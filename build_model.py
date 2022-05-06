# -*- coding: utf-8 -*-
import os, sys
import glob
import subprocess
import shapeworks as sw
sys.path.append(f"{SHAPEWORKS_DIR}/Examples/Python")
import OptimizeUtils
import AnalyzeUtils
import numpy as np
from pathlib import Path
from SSMUtils import create_shapeworks_projectsheet, create_excel_for_list
import math


INPUT_SEGMENTATION_DIR = '/home/sci/nawazish.khan/PML-Project-data/data/training/label'
MODELS_WORKING_DIR = '/home/sci/nawazish.khan/PML-Project/'
SHAPEWORKS_DIR = '/home/sci/nawazish.khan/ShapeWorks/'


class SSMProject:
    def __init__(self, **kwargs):

        self.create_folders = kwargs.get('create_folders', True)
        self.files_to_exclude = kwargs.get('seg_files_to_exclude', []) 

        self.segmentation_files = [] # Segmentations array
        self.meshes = [] # Meshes array  (points to original meshes before grooming step and later points to groomed meshes after Grooming step)
        self.shape_names = [] # subject names

        # Directories path
        self.project_dir = f"{MODELS_WORKING_DIR}/{self.ct_type}"
        print(f"Project Dir set to - {self.project_dir}")
        self.segmentations_dir = f"{self.project_dir}/segmentation_images/"
        self.meshes_dir = f"{self.project_dir}/meshes/"
        self.mesh_groom_dir = f"{self.meshes_dir}/groomed/"
        self.eval_out_dir = f"{self.meshes_dir}/output/"  # Output Directory for Evaluation metrics
        self.reconstructed_meshes_dir = f"{self.eval_out_dir}/reconstructed_meshes/"
        self.reconstruct_meshes_with_distance_dir = f"{self.eval_out_dir}/reconstructed_meshes_with_distance/"
        self.reference_warp_mesh_path = ""
        self.particles_dir = f"{self.project_dir}/{self.vertebra_type}_model_particles/"

        # Point Files 
        self.local_point_files = []
        self.world_point_files = []

        # Transforms
        self.rigid_transforms = []
        self.transforms = [] # final transformation matrix

        # Mesh Files
        self.mesh_files = []  #  Stores path to mesh files (points to original meshes before grooming step)
        self.groomed_mesh_files = [] # points to groomed meshes after Grooming step (transformation)
        self.reconstructed_mesh_files = []  # Stores path to reconstructed mesh files
        self.optimization_params = None

        # Studio Project file
        self.spreadsheet_fn = ""        

        if self.create_folders:
            if not os.path.exists(self.project_dir):
                os.makedirs(self.project_dir)
            if not os.path.exists(self.segmentations_dir):
                os.makedirs(self.segmentations_dir)
            if not os.path.exists(self.meshes_dir):
                os.makedirs(self.meshes_dir)
            if not os.path.exists(self.mesh_groom_dir):
                os.makedirs(self.mesh_groom_dir)
            if not os.path.exists(self.eval_out_dir):
                os.makedirs(self.eval_out_dir)
            if not os.path.exists(self.reconstructed_meshes_dir):
                os.makedirs(self.reconstructed_meshes_dir)
            if not os.path.exists(self.particles_dir):
                os.makedirs(self.particles_dir)

    
    def load_segmentations(self):
        print('----------Loading Segmentation list--------')
        parsed_sheet = create_shapeworks_projectsheet()
        seg_files_list = parsed_sheet[f"segmentation_{self.ct_type}"]

        for idx, seg_file_name in enumerate(seg_files_list):
            seg_name = Path(seg_file_name).stem
            seg_name = seg_name[0:len(seg_name)-3]
            if seg_name in self.files_to_exclude:
                print('Excluded shape: {}'.format(seg_name))
                continue
            self.shape_names.append(seg_name)
            self.segmentation_files.append(seg_file_name)

        idx_sorted = [self.shape_names.index(x) for x in sorted(self.shape_names)]
        self.shape_names = [self.shape_names[idx] for idx in idx_sorted]
        self.segmentation_files = [self.segmentation_files[idx] for idx in idx_sorted]

        print(f'Selected {len(self.segmentation_files)} CT samples')
        print('\n'.join(self.shape_names))

        # Compute iso spacing for shape cohort
        sp_x = np.array(parsed_sheet['CT_Resolution_spY'])
        sp_z = np.array(parsed_sheet['CT_Resolution_slice_thickness'])
        a = round(np.mean(sp_x), 2)
        b = round(np.mean(sp_z), 2)
        iso_val = min(a, b)
        self.iso_spacing = [iso_val, iso_val, iso_val]
        print(f'Iso spacing set = {self.iso_spacing}')

    
    def convert_to_meshes(self, specific_path_folder=None):
        """
            This function performs few grooming steps to the segmentation images and converts them to meshes
        """
        self.meshes = []
        for idx, seg_file in enumerate(self.segmentation_files):
            print(f"Converting {self.shape_names[idx]}")

            seg = sw.Image(seg_file)
            seg.antialias(15)
            seg.resample(self.iso_spacing, sw.InterpolationType.Linear)
            seg.computeDT()
            seg.gaussianBlur(1.0)
            mesh = seg.toMesh(0.0)  # Get iso surface
            mesh.fillHoles()
            # remesh_percent = int(0.75 * mesh.numPoints())
            # Retaining 75 % of vertices, anything below 65% - 68% leads to non-manifold edges(noted from expts)
            # mesh.remeshPercent(remesh_percent, 1.0)
            mesh.remeshPercent(percentage=75, adaptivity=1.0)  # Perform ACVD Remeshing
            self.meshes.append(mesh)
        # Save ACVD Remeshed Meshes
        meshes_dir = self.meshes_dir
        if specific_path_folder:
            os.makedirs(specific_path_folder, exist_ok=True)
            meshes_dir = specific_path_folder
        self.mesh_files = sw.utils.save_meshes(meshes_dir, self.meshes, self.shape_names, extension='vtk', compressed=False, verbose=True) # stores mesh paths
        print('----------Segmentation Images converted to meshes and loaded-----')

    def load_mesh_files_from_dir(self, groomed=False):
        """
            This function loads the meshes from the meshes directory (in case we want to start our project directly from existing meshes)
        """
        self.mesh_files = sorted(glob.glob(f'{self.meshes_dir}/*.vtk'))
        mesh_files_ = self.mesh_files
        if groomed:
            self.groomed_mesh_files = sorted(glob.glob(f"{self.mesh_groom_dir}/*.vtk"))
            mesh_files_ = self.groomed_mesh_files
        print(f"-----Found {len(self.mesh_files)} Meshes and {len(self.groomed_mesh_files)} Groomed Meshes-----")
        self.meshes = []
        self.shape_names = []
        # Load shape names as well
        print(f'Excluded shapes are {self.files_to_exclude}')
        for idx, mesh_file_name in enumerate(mesh_files_):
            mesh_name_ = Path(mesh_file_name).stem
            mesh_name_clean = mesh_name_.split('msk_')
            mesh_name = mesh_name_clean[0]+'msk'
            if mesh_name in self.files_to_exclude:
                print('Excluded shape: {}'.format(mesh_name))
                self.mesh_files.pop(idx)
                continue
            shape_mesh = sw.Mesh(mesh_file_name)
            self.meshes.append(shape_mesh)
            self.shape_names.append(mesh_name)
            print(f'Loaded -- {idx} -- {mesh_name}')
        if len(self.meshes) == 0:
            return False
        return True

    def groom_meshes(self, groom_eos=False):
        """
            Apply Common Grooming Steps to meshes
        """
        if len(self.meshes) == 0:
            if not self.load_mesh_files_from_dir(groomed=False):
                raise ValueError('Mesh files not present in dir')

        print('----Grooming Meshes-----')
        for idx, mesh in enumerate(self.meshes):
            mesh.smooth(iterations=2, relaxation=1.0)
            print(f'Done Smoothing--- {self.shape_names[idx]}')

        self.rigid_transform_meshes()
        self.save_transform()
        print('-----Rigid Transformation done')
        print("------Writing groomed meshes----")
        self.groomed_mesh_files = sorted(sw.utils.save_meshes(self.mesh_groom_dir, self.meshes, self.shape_names, extension='vtk', compressed=False, verbose=True))

    def save_transform(self):
        self.transforms = self.rigid_transforms
        # self.transforms = []
        # for translation, rigid_transform in zip(self.translations, self.rigid_transforms):
        #     self.transforms.append(np.matmul(rigid_transform, translation))

    def rigid_transform_meshes(self):
        # Find ref mesh and save it
        ref_index = sw.find_reference_mesh_index(self.meshes)
        self.ref_mesh = self.meshes[ref_index].copy().write(self.eval_out_dir + '/rigid_reference.vtk')
        ref_name = self.shape_names[ref_index]
        print("\nReference found: " + ref_name)
        for mesh in self.meshes:
            # compute rigid transformation
            rigid_transform = mesh.createTransform(self.ref_mesh, sw.Mesh.AlignmentType.Rigid, 100)
            # Apply rigid transform
            self.rigid_transforms.append(rigid_transform)
            # mesh.applyTransform(rigid_transform)

    def optimize(self, optimization_params, verbosity_val=0):
        """
            Given optimization params(dict), this function performs the optimization
        """
        if optimization_params is None:
            raise ValueError('Optimization Params must be specified')
        print('-------Running Optimization-------')
        if len(self.mesh_files) == 0:
            print('Found 0 mesh files currently, trying to load grooming files from directory')
            if not self.load_mesh_files_from_dir(groomed=True):
                raise ValueError('Mesh files must be specified')
        optimization_params["verbosity"] = verbosity_val
        [self.local_point_files, self.world_point_files] = OptimizeUtils.runShapeWorksOptimize(self.particles_dir, self.mesh_files, optimization_params)

    def build_project_file(self, optimization_params, verbosity_val=0):
        """
        Performs the optimization and generate shapeworks project file

        Args:
            optimization_params (_type_): _description_
        """
        if optimization_params is None:
            raise ValueError('Optimization Params must be specified')
        print('-------Building Project File-------')
        if len(self.groomed_mesh_files) == 0:
            print('Found 0 mesh files currently, trying to load grooming files from directory')
            if not self.load_mesh_files_from_dir(groomed=True):
                raise ValueError('Mesh files must be specified')
        # create subjects for shapes
        project_location = self.project_dir

        project_subjects = []
        for i in range(len(self.groomed_mesh_files)):
            subject = sw.Subject()
            subject.set_number_of_domains(1)
            rel_mesh_files = sw.utils.get_relative_paths([self.mesh_files[i]], project_location)
            subject.set_original_filenames(rel_mesh_files)
            rel_groom_files = sw.utils.get_relative_paths([self.groomed_mesh_files[i]], project_location)
            subject.set_groomed_filenames(rel_groom_files)
            transform = [ self.transforms[i].flatten() ]
            subject.set_groomed_transforms(transform)
            project_subjects.append(subject)
        
        # Set project
        shapeworks_project = sw.Project()
        shapeworks_project.set_subjects(project_subjects)

        # Initialize Optimization params
        parameters = sw.Parameters()
        optimization_params.pop('excluded_shapes')
        optimization_params["verbosity"] = verbosity_val
        for key in optimization_params:
            parameters.set(key, sw.Variant([optimization_params[key]]))
        parameters.set("domain_type", sw.Variant('mesh'))
        shapeworks_project.set_parameters("optimize", parameters)

        self.spreadsheet_fn = f"{project_location}/{self.ct_type}_model.xlsx"
        shapeworks_project.save(self.spreadsheet_fn)
        print(f"ShapeWorks Project saved at --- {self.spreadsheet_fn} ")

    def run_optimize(self):
        # Run optimization
        print('-----Running Optimization-----')
        optimize_cmd = ('shapeworks optimize --name ' + self.spreadsheet_fn).split()
        subprocess.check_call(optimize_cmd)

    def load_point_files(self, particle_type):
        """
            This function loads particle files for Analysis
        """
        print(f"{self.particles_dir}/*_{particle_type}.particles")
        if particle_type == 'local':
            self.local_point_files = sorted(glob.glob(f"{self.particles_dir}/*_{particle_type}.particles"))
            if len(self.local_point_files) == 0:
                return False
        else:
            self.world_point_files = sorted(glob.glob(f"{self.particles_dir}/*_{particle_type}.particles"))
            if len(self.world_point_files) == 0:
                return False
        return True

    def analyze_model(self):
        """
            Creates the analysis file and launches ShapeWorks Studio
        """
        if len(self.mesh_files) == 0:
            if not self.load_mesh_files_from_dir(groomed=True):
                raise ValueError('Mesh files must be specified')
        if len(self.local_point_files) == 0:
            if not self.load_point_files('local'):
                raise ValueError('Local Files must be specified')
        if len(self.world_point_files) == 0:
            if not self.load_point_files('world'):
                raise ValueError('Worlds Files must be specified')

        # Prepare analysis XML
        analyze_xml_fn = f"{self.particles_dir}/{self.ct_type}_analyze.xml"
        AnalyzeUtils.create_analyze_xml(analyze_xml_fn, self.mesh_files, self.local_point_files, self.world_point_files)
        AnalyzeUtils.launch_shapeworks_studio(analyze_xml_fn)
        print(f'-----Analysis for {ct_type} done -----')

    def analyze_studio(self):
        analyze_cmd = ('ShapeWorksStudio ' + self.spreadsheet_fn).split()
        subprocess.check_call(analyze_cmd)

    def compute_evaluation_params(self):
        """
            Explicitly compute, save and plot evaluation params(Compactness, Specificity, Generalization)
            Note that - this function can be used independently for any SSM model, provided that the Particle files exist as per the directory structure mentioned in the top comment.

        """
        if len(self.world_point_files) == 0:
            if not self.load_point_files('world'):
                raise ValueError('World Point Files must be specified')
        particle_sys = sw.ParticleSystem(self.world_point_files)
        print('-----Point Files loaded, Now computing Evaluation metrics-------')

        compactness = sw.ShapeEvaluation.ComputeFullCompactness(particleSystem=particle_sys)
        np.savetxt(f"{self.eval_out_dir}/Evaluation.txt", compactness)
        sw.plot.plot_mode_line(self.eval_out_dir,'Evaluation.txt',"Compactness","Explained Variance")
        print('---compactness plotted-----')

        print('-----Computing Generalization-----')
        generalization = sw.ShapeEvaluation.ComputeFullGeneralization(particleSystem=particle_sys)
        np.savetxt(f"{self.eval_out_dir}/Generalization.txt", generalization)
        sw.plot.plot_mode_line(self.eval_out_dir,'Generalization.txt',"Generalization","Generalization")
        print('---generalization plotted-----')

        print('-----Computing Specificity-----')
        specificity = sw.ShapeEvaluation.ComputeFullSpecificity(particleSystem=particle_sys)
        np.savetxt(f"{self.eval_out_dir}/Specificity.txt", specificity)
        sw.plot.plot_mode_line(self.eval_out_dir,'Specificity.txt',"Specificity","Specificity")
        print('---specificity plotted-----')

    def reconstruct_meshes(self):
        """
            Reconstruct meshes(for all subjects in the cohort) from the Optimized Set of Point files
            Needs:  Mean Points - Mean set of Particles of the SSM
                    Reference Mesh - Median Mesh of the cohort
                    Target/Moving Points - Particle files for the subjects

        """
        self.compute_median_shape()

        if len(self.local_point_files) == 0:
            if not self.load_point_files('local'):
                raise ValueError('Local Point Files must be specified')

        particle_sys = sw.ParticleSystem(self.local_point_files)
        median_particle = particle_sys.ShapeAsPointSet(self.median_shape_idx)
        median_mesh = self.meshes[self.median_shape_idx]

        warping = sw.MeshWarper()
        warping.generateWarp(median_mesh, median_particle)
        if warping.hasBadParticles():
            print('Warping has bad particle!')

        warping.getReferenceMesh().write(f"{self.eval_out_dir}/warp_ref_mesh.vtk")

        # Save the mean model without inserted triangles
        clean_shape = sw.MeshWarper.prepareMesh(median_mesh)
        nb_vertices_clean_shape = len(clean_shape.points())
        warp_matrix = warping.getWarpMatrix()[0:nb_vertices_clean_shape, :]  # Truncated matrix (without inserted vertices)

        vertices = np.matmul(warp_matrix, median_particle)
        mean_mesh = sw.Mesh(vertices, clean_shape.faces())
        self.mean_model_path = f"{self.eval_out_dir}/mean_mesh.vtk"
        mean_mesh.write(self.mean_model_path)
        mean_mesh.write(f"{self.eval_out_dir}/mean_mesh.stl")
        np.savetxt(f"{self.eval_out_dir}/mean_mesh.particles", warping.getReferenceParticles())
        np.save(f"{self.eval_out_dir}/warp_matrix", warp_matrix)

        # Warp the mesh and save it. Remove inserted vertices to have clean meshing (without small triangle added for warping)
        for idx, point_file in enumerate(self.local_point_files):
            mesh_name = Path(point_file).stem

            vertices = np.matmul(warp_matrix, particle_sys.ShapeAsPointSet(idx))
            mesh = sw.Mesh(vertices, clean_shape.faces())
            mesh.write(str(self.reconstructed_meshes_dir / Path(mesh_name+'.vtk')))

        self.reconstructed_mesh_files = sorted(glob.glob(f"{self.reconstructed_meshes_dir}/*.vtk"))
        if len(self.reconstructed_mesh_files) == 0:
            return False
        return True

    def build_shape_matrix(self):
        """
            Builds the shape matrix from the set of particles (world)
        """
        if len(self.world_point_files) == 0:
            if not self.load_point_files('world'):
                raise ValueError('World Point Files must be specified')
        particle_sys = sw.ParticleSystem(self.world_point_files)
        return particle_sys.Particles()

    def compute_median_shape(self):
        """
            Finds the median shape of the cohort, the one with Minimum L1 Norm
        """
        self.median_shape_idx = -1
        min_sum = 1e10
        if len(self.mesh_files) == 0:
            if not self.load_mesh_files_from_dir(groomed=True):
                raise ValueError('Mesh files must be specified')

        self.shape_matrix = self.build_shape_matrix()

        num_shapes = self.shape_matrix.shape[1]
        for i in range(0, num_shapes):
            cur_sum = 0.0
            for j in range(0, num_shapes):
                if i != j:
                    cur_sum += self.find_norm(i, j)
            if cur_sum < min_sum:
                min_sum = cur_sum
                self.median_shape_idx = i

        if self.median_shape_idx == -1:
            raise ValueError('Median shape not found for Reconstruction, Cannot proceed further with Mesh Warping')

        ref_median_shape__name = self.shape_names[self.median_shape_idx]
        print("\nWarping uses the median reference shape: " + ref_median_shape__name)
        self.reference_warp_mesh_path = f"{self.eval_out_dir}/reference_mesh.vtk"
        self.meshes[self.median_shape_idx].copy().write(self.reference_warp_mesh_path)

        return True

    def find_norm(self, a, b):
        """
            Utility function to compute norm between two shape vectors 'a' and 'b' of the cohort
        """
        norm = 0.0
        for i in range(0, self.shape_matrix.shape[0]):
            norm += (math.fabs(self.shape_matrix[i, a] - self.shape_matrix[i, b]))
        return norm

    def compute_surface_surface_distance_metric(self):
        """
            This function computes the surface-surface distance between Reconstructed meshes and Groomed Meshes.
            Note that - this function can be used independently for any SSM model, provided that the Particle files and Groomed Meshes exist as per the directory structure mentioned in the top comment.
        """
        # First load point files, if not loaded until now
        if len(self.world_point_files) == 0:
            if not self.load_point_files('world'):
                raise ValueError('World Point Files must be specified')
        if len(self.local_point_files) == 0:
            if not self.load_point_files('local'):
                raise ValueError('Local Point Files must be specified')
        if len(self.reconstructed_mesh_files) == 0:
            if not self.reconstruct_meshes():
                raise ValueError("Reconstructed Meshes not build/available")
        if not os.path.exists(self.reconstruct_meshes_with_distance_dir):
            os.makedirs(self.reconstruct_meshes_with_distance_dir)
        print('-----Mesh Warping done, Now computing Surface-Surface distance metric-----')

        errors = []

        idx = 0
        for groomed_mesh_file, reconst_mesh_file in zip(self.mesh_files, self.reconstructed_mesh_files):
            groomed_mesh = sw.Mesh(groomed_mesh_file)
            reconst_mesh = sw.Mesh(reconst_mesh_file)

            if len(errors) == 0:
                errors = np.ones(shape=(len(self.mesh_files), len(reconst_mesh.points()))) * np.NaN

            dists_and_indexes = reconst_mesh.distance(target=groomed_mesh, method=sw.Mesh.DistanceMethod.PointToCell)
            distances = np.array(dists_and_indexes[0])

            out_file_name = f"{self.reconstruct_meshes_with_distance_dir}/{self.shape_names[idx]}.vtk"
            out_file_name = str(Path(out_file_name))  # For windows path
            out_file_name = out_file_name.replace('\\', '/')
            out_file_name = out_file_name.replace('//', '/')

            reconst_mesh.setField('distance', distances, sw.Mesh.Point)
            reconst_mesh.write(out_file_name)

            errors[idx, :] = distances
            idx += 1

        print('---------Reconstructed meshes written with distance----------')

        print('Global error (Reconstructed vs. groomed) mean={} rms={} max={}'.format(np.mean(errors), np.sqrt(np.mean(errors**2)), np.max(errors)))

        mean_mesh = sw.Mesh(self.mean_model_path)
        mean_mesh.setField('mean_error_map', np.mean(errors, axis=0), sw.Mesh.Point)
        mean_mesh.setField('rms_error_map', np.sqrt(np.mean(errors**2, axis=0)), sw.Mesh.Point)
        mean_mesh.setField('max_error_map', np.max(errors, axis=0), sw.Mesh.Point)
        mean_mesh.write(self.mean_model_path)

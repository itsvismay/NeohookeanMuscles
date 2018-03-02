#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Plame strain triangular element tris
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>

#include <igl/viewer/Viewer.h>
#include <igl/readPLY.h>
#include <json.hpp>

using json = nlohmann::json;

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

//typedef scene
typedef PhysicalSystemFEM<double, MuscleTri> FEMPlaneStrainTri;

typedef World<double, std::tuple<FEMPlaneStrainTri *>, std::tuple<ForceSpringFEMParticle<double> *>, std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplicitLinear<double, AssemblerEigenSparseMatrix<double>, AssemblerEigenVector<double> > MyTimeStepper;

Eigen::MatrixXd V;
Eigen::MatrixXi F;

Eigen::MatrixXd getCurrentVertPositions(MyWorld &world, FEMPlaneStrainTri *tets) {
    // Eigen::Map<Eigen::MatrixXd> q(mapStateEigen<0>(world).data(), V.cols(), V.rows()); // Get displacements only
    auto q = mapDOFEigen(tets->getQ(), world);
    Eigen::Map<Eigen::MatrixXd> dV(q.data(), V.cols(), V.rows()); // Get displacements only

    return V + dV.transpose();
}


int main(int argc, char **argv) {
    std::cout<<"2d Neohookean Muscle\n";
    std::ifstream i("../../input/input2d.json");
	json j_input;
	i >> j_input;
    //Setup Physics
    MyWorld world;


    //simple square subdivided into two triangles
    igl::readPLY(j_input["mesh_file"], V, F);


    FEMPlaneStrainTri *tris = new FEMPlaneStrainTri(V,F);

    Eigen::Vector3d grav(0,0,0);
    for(auto element: tris->getImpl().getElements())
    {
    	element->EnergyKineticNonLumped::setDensity(100.0);//1000.0);
        element->setParameters(j_input["youngs"], j_input["poissons"]);
        element->setGravity(grav);
    }
    Eigen::Vector3d maxs = getMaxXYZ(world, tris);
    Eigen::Vector3d mins = getMinXYZ(world, tris);
    double t = j_input["thresh"];
    Eigen::Vector3d thresh = mins + t*(maxs - mins);
    Eigen::Vector3d fibre_dir(j_input["fibre_dir"][0], j_input["fibre_dir"][1], j_input["fibre_dir"][2]);
    int axis = j_input["axis"];
    for(auto element: tris->getImpl().getElements()) 
    {
        int i0 = (element->EnergyKineticNonLumped::getQDofs())[0]->getLocalId();
        int i1 = (element->EnergyKineticNonLumped::getQDofs())[1]->getLocalId();
        int i2 = (element->EnergyKineticNonLumped::getQDofs())[2]->getLocalId();

        if(tris->getImpl().getV().row(i0/2)(axis) > thresh(axis)||
            tris->getImpl().getV().row(i1/2)(axis)> thresh(axis)||
            tris->getImpl().getV().row(i2/2)(axis)> thresh(axis))
        {
            element->setMuscleParameters(j_input["fibre_mag"], fibre_dir);
        }

    }


    world.addSystem(tris);
    fixDisplacementMax(world, tris, 1);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    

    auto q = mapStateEigen(world);
    q.setZero();

    MyTimeStepper stepper(0.01);


     /** libigl display stuff **/
    igl::viewer::Viewer viewer;

    viewer.callback_pre_draw = [&](igl::viewer::Viewer& viewer)
    {
    	if(viewer.core.is_animating)
    	{
    		stepper.step(world);

    		Eigen::MatrixXd newV = getCurrentVertPositions(world, tris);
    		viewer.data.set_vertices(newV);
    		viewer.data.compute_normals();
    	}
    	return false;
    };

    viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = false;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;

    viewer.launch();
    return EXIT_SUCCESS;



}
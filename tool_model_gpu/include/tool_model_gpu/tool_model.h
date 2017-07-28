#ifndef TOOL_MODEL_H
#define TOOL_MODEL_H

#include <vector>
#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <string>
#include <cstring>

#include <glm/glm.hpp>

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <cwru_opencv_common/projective_geometry.h>
#include <ros/package.h>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>


class ToolModel {

private:

/**
 * @brief tool_model_pkg gives the path finding all the tool body parts inside the workspace
 */
std::string tool_model_pkg;

public:
    struct toolModel {
        cv::Matx<double, 3, 1> tvec_cyl;    //cylinder translation vector
        cv::Matx<double, 3, 1> rvec_cyl;    //cylinder rotation vector

        cv::Matx<double, 3, 1> tvec_elp;    //ellipse translation vector
        cv::Matx<double, 3, 1> rvec_elp;    //ellipse rotation vector

        cv::Matx<double, 3, 1> tvec_grip1;    //gripper1 translation vector
        cv::Matx<double, 3, 1> rvec_grip1;    //gripper1 rotation vector

        cv::Matx<double, 3, 1> tvec_grip2;    //gripper2 translation vector
        cv::Matx<double, 3, 1> rvec_grip2;    //gripper2 rotation vector

        //built in operator that handles copying the toolModel in efficient way
        toolModel &operator=(const toolModel src) {

            for (int i(0); i < 3; i++) {

                this->tvec_cyl(i) = src.tvec_cyl(i);
                this->rvec_cyl(i) = src.rvec_cyl(i);

                this->tvec_elp(i) = src.tvec_elp(i);
                this->rvec_elp(i) = src.rvec_elp(i);

                this->tvec_grip1(i) = src.tvec_grip1(i);
                this->rvec_grip1(i) = src.rvec_grip1(i);

                this->tvec_grip2(i) = src.tvec_grip2(i);
                this->rvec_grip2(i) = src.rvec_grip2(i);

            }
            return *this;
        }

        // toolModel constructor, creates an empty toolModel
        toolModel() {
            for (int i(0); i < 3; i++) {
                this->tvec_cyl(i) = 0.0;
                this->rvec_cyl(i) = 0.0;

                this->tvec_elp(i) = 0.0;
                this->rvec_elp(i) = 0.0;

                this->tvec_grip1(i) = 0.0;
                this->rvec_grip1(i) = 0.0;

                this->tvec_grip2(i) = 0.0;
                this->rvec_grip2(i) = 0.0;

            }
        }

    };   //end struct

    /******************the model points and vecs********************/
    std::vector<glm::vec3> body_vertices;
    std::vector<glm::vec3> ellipse_vertices;
    std::vector<glm::vec3> griper1_vertices;
    std::vector<glm::vec3> griper2_vertices;

    std::vector<glm::vec3> body_Vnormal;    //vertex normal, for the computation of the silhouette
    std::vector<glm::vec3> ellipse_Vnormal;
    std::vector<glm::vec3> griper1_Vnormal;
    std::vector<glm::vec3> griper2_Vnormal;

    std::vector<cv::Point3d> body_Vpts;     ////to save time from converting
    std::vector<cv::Point3d> ellipse_Vpts;
    std::vector<cv::Point3d> griper1_Vpts;
    std::vector<cv::Point3d> griper2_Vpts;

    std::vector<cv::Point3d> body_Npts;    //vertex normal, for the computation of the silhouette
    std::vector<cv::Point3d> ellipse_Npts;
    std::vector<cv::Point3d> griper1_Npts;
    std::vector<cv::Point3d> griper2_Npts;

    cv::Mat body_Vmat;
    cv::Mat ellipse_Vmat;
    cv::Mat gripper1_Vmat;
    cv::Mat gripper2_Vmat;

    cv::Mat body_Nmat;
    cv::Mat ellipse_Nmat;
    cv::Mat gripper1_Nmat;
    cv::Mat gripper2_Nmat;

    /**************************Faces information********************************/
    std::vector<int> body_faces;
    std::vector<int> ellipse_faces;
    std::vector<int> griper1_faces;
    std::vector<int> griper2_faces;

    std::vector<int> body_neighbors;
    std::vector<int> ellipse_neighbors;
    std::vector<int> griper1_neighbors;
    std::vector<int> griper2_neighbors;

    cv::Mat bodyFace_normal;
    cv::Mat ellipseFace_normal;
    cv::Mat gripper1Face_normal;
    cv::Mat gripper2Face_normal;

    cv::Mat bodyFace_centroid;
    cv::Mat ellipseFace_centroid;
    cv::Mat gripper1Face_centroid;
    cv::Mat gripper2Face_centroid;

    /**************************** offset for modify model **************************************/
    double offset_body;
    double offset_ellipse; // all in meters
    double offset_gripper; //

    /************** basic funcs ****************/
    ToolModel();  //constructor, TODO: we need this to be the camera extrinsic param relative to the tool frame

    double randomNumber(double stdev, double mean);
    double randomNum(double min, double max);

    void
    load_model_vertices(const char *path, std::vector<glm::vec3> &out_vertices, std::vector<glm::vec3> &vertex_normal,
                        std::vector<int> &out_faces, std::vector<int> &neighbor_faces);

    void modify_model_(std::vector<glm::vec3> &input_vertices, std::vector<glm::vec3> &input_Vnormal,
                       std::vector<cv::Point3d> &input_Vpts, std::vector<cv::Point3d> &input_Npts, double &offset,
                       cv::Mat &input_Vmat, cv::Mat &input_Nmat);//there are offsets when loading the model convert

    toolModel setRandomConfig(const toolModel &seeds, const double &theta_cylinder, const double &theta_oval, const double &theta_open, double &step);
    ToolModel::toolModel gaussianSampling(const toolModel &max_pose, double &step);

    /****can use the best match tool pose generated by computeModelPose(), then render the ellipse part ****/

    /// compute the pose for the tool model based on the CYLINDER pose, return the pose within inputModel
    void computeEllipsePose(toolModel &inputmodel, const double &theta_ellipse, const double &theta_grip_1,
                          const double &theta_grip_2);

    void computeRandomPose(const toolModel &seed_pose, toolModel &inputModel, const double &theta_tool, const double &theta_grip_1,
                                      const double &theta_grip_2);
    //cam view need to be modified
    void renderTool(cv::Mat &image, const toolModel &tool, cv::Mat &CamMat, const cv::Mat &P);

    ///measurement for perception model
    float calculateMatchingScore(cv::Mat &toolImage, const cv::Mat &segmentedImage);
    float calculateChamferScore(cv::Mat &toolImage, const cv::Mat &segmentedImage);

    /**********************math computation*******************/
    cv::Point3d crossProduct(cv::Point3d &vec1, cv::Point3d &vec2);

    double dotProduct(cv::Point3d &vec1, cv::Point3d &vec2);

    cv::Point3d Normalize(cv::Point3d &vec1);

    void ConvertInchtoMeters(std::vector<cv::Point3d> &input_vertices);

    void Convert_glTocv_pts(std::vector<glm::vec3> &input_vertices, std::vector<cv::Point3d> &out_vertices);

    cv::Mat computeSkew(cv::Mat &w);

    void computeInvSE(const cv::Mat &inputMat, cv::Mat &outputMat);
    /****************Face editing and info**********************/
    cv::Point3d FindFaceNormal(cv::Point3d &input_v1, cv::Point3d &input_v2, cv::Point3d &input_v3,
                               cv::Point3d &input_n1, cv::Point3d &input_n2, cv::Point3d &input_n3);

    int Compare_vertex(std::vector<int> &vec1, std::vector<int> &vec2, std::vector<int> &match_vec);

    void getFaceInfo(const std::vector<int> &input_faces, const std::vector<cv::Point3d> &input_vertices,
                     const std::vector<cv::Point3d> &input_Vnormal, cv::Mat &face_normals, cv::Mat &face_centroids);

    /*************camera transforms************************/
    cv::Mat camTransformMats(cv::Mat &cam_mat, cv::Mat &input_mat);


    /**
     * GPU functions
     */
    void ComputeSilhouetteGPU(std::vector<int> &input_faces, std::vector<int> &neighbor_faces,
                                         const cv::Mat &input_Vmat, const cv::Mat &input_Nmat,
                                         cv::Mat &CamMat, cv::Mat &image, const cv::Mat &rvec, const cv::Mat &tvec,
                                         const cv::Mat &P);

    cv::Point2d reproject(const cv::Mat &point, const cv::Mat &P);

};
__global__ void renderingkernel(cv::cuda::PtrStepSzf new_Vertices, cv::cuda::PtrStepSzf device_new_Normals,  int  * input_faces,
                                            int * neighbor_faces, int * ind);
//TODO add output as input

__device__ double dotProductGPU(double3 &vec1, double3 &vec2);

__device__ cv::Point3d convert_MattoPts(cv::cuda::GpuMat &input_Mat);

__device__ double3 FindFaceNormalGPU(double3 &input_v1, double3 &input_v2, double3 &input_v3,
                double3 &input_n1, double3 &input_n2, double3 &input_n3);

__device__ double3 crossProductGPU(double3 &vec1, double3 &vec2);
#endif
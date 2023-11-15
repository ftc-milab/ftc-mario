#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    // Open the video file
    cv::VideoCapture cap("/home/mario/ftc/FTC-2024-data/Train/train.mp4");

    // Check if the video file is opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return -1;
    }

    // Get the frames per second (fps) of the video
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Get the width and height of the video frames
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Create a VideoWriter object to save the images
    cv::VideoWriter video_writer("output_images/video_frame_%04d.jpg",
                                  cv::VideoWriter::fourcc('J', 'P', 'E', 'G'),
                                  fps,
                                  cv::Size(frame_width, frame_height));

    // Check if the VideoWriter object is opened successfully
    if (!video_writer.isOpened()) {
        std::cerr << "Error opening VideoWriter." << std::endl;
        return -1;
    }

    // Loop through the frames of the video
    cv::Mat frame;
    int frame_number = 0;
    while (true) {
        // Read the next frame from the video
        cap >> frame;

        // Break the loop if the video is finished
        if (frame.empty()) {
            break;
        }

        // Save the current frame as a JPEG image
        std::string filename = "/home/mario/ftc/original_cpp/images/frame" + std::to_string(frame_number) + ".jpg";
        cv::imwrite(filename, frame);

        // Write the frame to the VideoWriter (optional)
        video_writer.write(frame);

        // Increment the frame number
        frame_number++;
    }

    // Release the VideoCapture and VideoWriter objects
    cap.release();
    video_writer.release();

    return 0;
}




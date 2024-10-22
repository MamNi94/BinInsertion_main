import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import queue
import time

from tensorflow.keras.preprocessing import image
import tensorflow as tf

from roi_functions import  cut_region_between_hulls


# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print GPU details if available
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.list_physical_devices('GPU'):
        print("GPU Device:", gpu)

        
wall_model = tf.keras.models.load_model('models/walls/inception_wall_rect_224x224_v0_L2_val_accuracy_0.993_combined_data.h5')
frame_queue =queue.Queue(maxsize=1)

def capture_frames(pipeline, align, exit_event,temporal_filter,spatial_filter,hole_filling_filter ):
    global frame_queue
   
  
    try:
        while not exit_event.is_set():
      
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            #filtered_depth_frame = temporal_filter.process(depth_frame) 
            #filtered_depth_frame = spatial_filter.process(filtered_depth_frame)
            #filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)
            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(color_frame.get_data())

            
            
            
            # Put the latest frame into the queue
            if not frame_queue.full():
                frame_queue.put((color_image, depth_image))
            else:
                try:
                    frame_queue.get_nowait()  # Discard the old frame
                except queue.Empty:
                    pass
                frame_queue.put((color_image, depth_image))
                
    except RuntimeError as e:
        print(f"Capture stopped: {e}")
    finally:
        print("Capture thread finished.")
        
def detect_walls(color_image,masked_color_image, wall_model, number =1):
    factor_x = 224/1920
    factor_y = 224/1080
    input_wallcheck = cv2.resize(masked_color_image,None, fx = factor_x, fy =factor_y)
    #cv2.imshow('CNN input Wallcheck', input_wallcheck)
    img_array = image.img_to_array(input_wallcheck)

    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Normalize

    with tf.device('/GPU:0'):
        prediction = wall_model.predict(img_array)
    print(f'prediction {prediction[0]}')
    height, width = color_image.shape[:2]
    h = np.int0(height/2)
    w = np.int0(width/2)  
    
    if prediction[0] > 0.5:
        cv2.putText(color_image,f'Wall Check: ', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 3)
        cv2.putText(color_image,f'Passed', (w+130,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 0), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 3)
        check = True
    else:
        cv2.putText(color_image,f'Wall Check:', (w-60,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 3)
        cv2.putText(color_image,f'Failed', (w+130,h+60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 200), 3)
        cv2.putText(color_image,f'Confidence: {prediction[0]}', (w-60,h+100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 3)
        check = False
        
    return check

def bincontrol(frame_queue_main:queue.Queue,inserted_bins:queue.Queue,stop_flag:threading.Event):
    global frame_queue
    inserted_bins_count = 0
    image_count = 0
    box_detected_last_iteration = False
    wall_check = False
    
    try:
        # Your bincontrol logic here
        
        while not stop_flag.is_set():
           
            if not frame_queue.empty():
                # Process frames
           
               
                color_image = None
                color_image, depth_image = frame_queue.get()
                masked_color_image,cropped_image, hull,box,box_detected = cut_region_between_hulls(depth_image,color_image,min_depth = 0,max_depth = 0.8,  cut_rect= True, improved_bbox=False)


                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.25), cv2.COLORMAP_VIRIDIS)
                scale_factor = 0.7
                resized_depth_image = cv2.resize(depth_colormap,None,fx =  scale_factor,fy = scale_factor)
                cv2.imshow('depth image',resized_depth_image)
                if box_detected ==False and box_detected_last_iteration == True and wall_check == True:
                        inserted_bins_count +=1
                        inserted_bins.put(inserted_bins_count)
                        
                if box_detected == True:
                    wall_check = detect_walls(color_image,masked_color_image,wall_model,1)
                    #color_image = masked_color_image
                    cv2.imwrite(f'saved_images/positive_{image_count}.jpg', masked_color_image)
                    image_count +=1
                
                scale_factor = 0.4
                resized_image = cv2.resize(color_image,None,fx =  scale_factor,fy = scale_factor)
                
                box_detected_last_iteration = box_detected
                
                if not frame_queue_main.full():
                    frame_queue_main.put(resized_image)
               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_flag.set()
                    break

    finally:
        # Stop the pipeline here
        #pipeline.stop()
        cv2.destroyAllWindows()
        print('Bincontrol stopped')


#def main(pipeline,config, stop_event):
def main(frame_queue_main:queue.Queue,inserted_bins:queue.Queue, stop_event):  

    print('main started.....')
    # Initialize RealSense pipeline
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, framerate=30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, framerate=30)
    pipeline.start(config)
    
    # Create align object
    align_to = rs.stream.color
    align = rs.align(align_to)

     # 1. Temporal Filter
    temporal_filter = rs.temporal_filter()

    # 2. Spatial Filter
    spatial_filter = rs.spatial_filter()

    # 3. Hole Filling Filter
    hole_filling_filter = rs.hole_filling_filter()

    
    # Start the capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(pipeline, align, stop_event, temporal_filter,spatial_filter,hole_filling_filter), daemon=True)
    capture_thread.start()
    print('capture thread started')
    
    # Start the bincontrol thread
    bincontrol_thread = threading.Thread(target=bincontrol, args= (frame_queue_main,inserted_bins,stop_event,), daemon=True)
    bincontrol_thread.start()
    print('bincontrol thread started')

    try:
        while True:
            if stop_event.is_set():
                break
            time.sleep(0.1)  # Main loop can do other work or just wait

    finally:
        # Signal threads to stop
        stop_event.set()

        # Wait for threads to finish
        capture_thread.join()
        
        bincontrol_thread.join()

        # Stop the pipeline safely
        pipeline.stop()
        cv2.destroyAllWindows()
        print("All threads and pipeline stopped")
        
        
if __name__ == "__main__":
    #stop_flag = threading.Event()
    main()
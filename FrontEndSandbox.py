import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

import threading
import pyrealsense2 as rs
import queue
import GUIBincontrol
import cv2
import os

# Create the 'saved_images' directory if it doesn't exist
if not os.path.exists('saved_images'):
    os.makedirs('saved_images')


frame_queue_main = queue.Queue(maxsize = 1)
inserted_bins = queue.Queue(maxsize = 1)

stop_flag = threading.Event()

main_thread = None


def show_about():
    messagebox.showinfo("About", "This is a the Kardex BINcontrol app")
    

def start_bincontrol():
    
    global stop_flag
    global main_thread

    stop_flag.clear()
    main_thread = threading.Thread(target = GUIBincontrol.main, args = (frame_queue_main,inserted_bins,stop_flag,), daemon = True)
    main_thread.start()
    
def terminate_bincontrol():
    global stop_flag
    global main_thread
    stop_flag.set()
    if main_thread is not None:     
        main_thread.join()
        main_thread= None
       
        
def on_button_click():
    output_label.config(text=f"Welcome to KardexBINsertion!{i.get()}")
    i.set(i.get()+1)
    
def bad_button_click():
    output_label.config(text=f"Welcome to KardexBINsertion!{i.get()}")
    i.set(i.get()-1)
    

def exit_program():
    terminate_bincontrol() 
    print("program terminated")
    root.destroy()
    
def update_variable_display():
    """Update the display with the latest variable value."""
    if not inserted_bins.empty():
        value = inserted_bins.get()
        inserted_bins_label.config(text=f"Inserted Bins: {value}")  # Update label with the new value

    # Schedule the next update
    inserted_bins_label.after(100, update_variable_display) 
    

def update_frame():
    global stop_flag
    # Check if the frame queue is empty
    if not frame_queue_main.empty():
        color_image = frame_queue_main.get()  # Get the latest frame from the queue
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(color_image)  # Convert the frame to a PIL image
        imgtk = ImageTk.PhotoImage(image=img)  # Convert the PIL image to ImageTk format
        video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
        video_label.configure(image=imgtk)  # Update the label with the new image
    if stop_flag.is_set():
        # Clear the video label if the queue is empty
       video_label.configure(image='')  # Set the image to an empty string to clear it

    # Schedule the next frame update
    video_label.after(10, update_frame)
    

# Create the main window
root = tk.Tk()
root.title("BINsertion app alpha")
root.geometry("1600x900")

i = tk.IntVar()
i.set(0)
   
image_path = 'background_2.jpg'
image = Image.open(image_path)
new_size = (1600, 900)  # Set your desired dimensions here
resized_image = image.resize(new_size, Image.LANCZOS)
photo = ImageTk.PhotoImage(resized_image)



background_label = tk.Label(root, image = photo)
background_label.place(relwidth = 1, relheight = 1)

# Create a menu bar
menu_bar = tk.Menu(root)

# Create a File menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command( label="New")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Create a Help menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About", command=show_about)
menu_bar.add_cascade(label="Help", menu=help_menu)

# Display the menu
root.config(bg = 'orange',menu=menu_bar)


# Add a label to display output
#output_label = tk.Label(root, text="Click the button below!")
#output_label.place(x=20,y = 10)

video_label = tk.Label(root)
video_label.place(x=300, y=100)  # Position it appropriately

inserted_bins_label = tk.Label(root, text="Inserted Bins: 0",font=("Helvetica", 12))
inserted_bins_label.place(x=300, y=550)

# Add a button that triggers an action
#click_button = tk.Button(root, text="Click Me",  command=on_button_click)
#click_button.place(x = 50,y = 40)

#click_button = tk.Button(root, text="Dont Click Me!",  command=bad_button_click)
#click_button.place(x = 50,y = 80)

click_button = tk.Button(root, text="Bincontrol", bg = 'green', command=start_bincontrol)
click_button.place(x = 50,y = 120)

click_button = tk.Button(root, text="End Bincontrol", bg = 'orange', command=terminate_bincontrol)
click_button.place(x = 50,y = 160)

exit_button = tk.Button(root, text="Exit", width = 5, height = 2,bg = 'red', fg = 'white',command=exit_program)
exit_button.place(x = 1500, y = 830)

update_frame() 
update_variable_display()
# Run the application
root.mainloop()


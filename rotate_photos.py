import os
from PIL import Image, ExifTags

def rotate_image(image):
    """
    Checks an image's EXIF data for an Orientation tag and
    rotates the image to the proper orientation.
    """
    try:
        # Attempt to retrieve the image's EXIF data
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is not None:
        # Find the tag number for 'Orientation'
        orientation_tag = None
        for tag, value in ExifTags.TAGS.items():
            if value == 'Orientation':
                orientation_tag = tag
                break

        if orientation_tag is not None:
            orientation = exif.get(orientation_tag, None)

            # Depending on the orientation value, rotate accordingly:
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    return image

def process_folder(folder_path):
    """
    Processes all JPEG images in the given folder,
    rotating them based on their EXIF Orientation data,
    and saves the rotated images in a 'rotated' subfolder.
    """

    rotated_dir = os.path.join(folder_path, "rotated")
    if not os.path.exists(rotated_dir):
        os.makedirs(rotated_dir)

    # Loop over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    # Rotate the image based on its EXIF orientation tag
                    rotated_img = rotate_image(img)
                    # Save the rotated image
                    save_path = os.path.join(rotated_dir, filename)
                    rotated_img.save(save_path)
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":

    folder_dir = input("Enter the folder path containing your photos: ").strip()
    if os.path.isdir(folder_dir):
        process_folder(folder_dir)
    else:
        print("The folder path provided does not exist. Please check and try again.")
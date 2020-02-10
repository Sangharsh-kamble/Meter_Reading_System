import os
def directory():
    if not os.path.exists('MeterImages'):
        os.makedirs('MeterImages')
    root_path = 'MeterImages/'
    folders = ['allImages','screen_images','barcode_images']
    for folder in folders:
        try:
            if not os.path.exists(folder):
                os.mkdir(os.path.join(root_path,folder))
        except FileExistsError:
            pass
directory()

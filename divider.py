import os
import cv2
import numpy as np


def load_image(image_dir, row, col):
    filename = f"phlr{row:02d}c{col:02d}.tif"
    filepath = os.path.join(image_dir, filename)
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    else:
        return np.zeros((8574, 8574, 3), dtype=np.uint8)



def save_subtile(output_dir, subtile, global_row, global_col):
    filename = f"subtile_{global_row}_{global_col}.png"
    cv2.imwrite(os.path.join(output_dir, filename), subtile)


def stack_and_generate_subtiles(main_tile, right_tile, bottom_tile, bottom_right_tile,
                                 subtile_size, stride, output_dir):
    top_row = np.concatenate((main_tile, right_tile), axis=1) if right_tile is not None else main_tile
    bottom_row = np.concatenate((bottom_tile, bottom_right_tile), axis=1) if bottom_tile is not None else bottom_tile
    stacked_image = np.concatenate((top_row, bottom_row), axis=0) if bottom_row is not None else top_row

    stacked_h, stacked_w, _ = stacked_image.shape

    global_row = 69
    global_col = 0
    local_row = 0
    local_col = 0
    row_offset = (global_row * stride) % 8574
    col_offset =  (global_col * stride) % 8574
    m = 0
    while local_row < 23:
        y_start = stride * m + row_offset
        y_end = y_start + subtile_size

        if y_end > stacked_h:
            break

        n = 0
        while local_col < 23:
            x_start = stride * n + col_offset
            x_end = x_start + subtile_size

            if x_end > stacked_w:
                break

            subtile = stacked_image[y_start:y_end, x_start:x_end]
            save_subtile(output_dir, subtile, global_row, global_col)

            n += 1
            global_col += 1
            local_col +=1

        m += 1
        global_row += 1
        local_row += 1
        local_col = 0
        global_col = 0


def load_neighbors(image_directory, row, col):
    main_tile = load_image(image_directory, row, col)
    right_tile = load_image(image_directory, row, col + 1)
    bottom_tile = load_image(image_directory, row + 1, col)
    bottom_right_tile = load_image(image_directory, row + 1, col + 1)
    return main_tile, right_tile, bottom_tile, bottom_right_tile


def main():
    image_directory = "Philadelphia tiles"
    output_directory = "Philadelphia subtiles 512"
    subtile_size = 512
    stride = 384

    selected_row = 11
    selected_col = 13

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    main_tile, right_tile, bottom_tile, bottom_right_tile = load_neighbors(image_directory, selected_row, selected_col)

    if main_tile is not None:
        stack_and_generate_subtiles(
            main_tile=main_tile,
            right_tile=right_tile,
            bottom_tile=bottom_tile,
            bottom_right_tile=bottom_right_tile,
            subtile_size=subtile_size,
            stride=stride,
            output_dir=output_directory
        )


if __name__ == "__main__":
    main()
#     rename_images_in_directory_to_grid('Philadelphia subtiles 512')

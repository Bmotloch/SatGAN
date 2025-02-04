import cv2
import numpy as np
import os
import re


def extract_row_col_filename(filename):
    base_name = os.path.basename(filename)
    match = re.search(r"subtile_(\d+)_(\d+)", base_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        print(f"Filename does not match pattern: {filename}")
        return None, None


def find_neighbors_sub(y_index, x_index, tile_files, input_dir, mode='row'):
    neighbors = []
    for file in tile_files:
        file_y, file_x = extract_row_col_filename(file)
        if file_y is not None and file_x is not None:
            if mode == 'row' and file_y == y_index and file_x == x_index + 1:
                neighbors.append(os.path.join(input_dir, file))
            elif mode == 'col' and file_y == y_index + 1 and file_x == x_index:
                neighbors.append(os.path.join(input_dir, file))
    return neighbors


def match_features(des1, des2, method='BF', cross_check=True):
    if method == 'BF':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    elif method == 'FLANN':
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("Unsupported method. Use 'BF' or 'FLANN'.")

    matches = matcher.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)


def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        print(f"Insufficient matches for homography: {len(matches)} found.")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or not np.isfinite(H).all():
        print("Homography computation failed or returned invalid results.")
        return None, None

    return H, mask


def mask_images(img1, img2, mode):
    img1_masked = img1.copy()
    img2_masked = img2.copy()

    if mode == "row":
        img1_masked[:, :3 * img1.shape[1] // 4] = 0
        img2_masked[:, img2.shape[1] // 4:] = 0
    elif mode == "col":
        img1_masked[:3 * img1.shape[0] // 4, :] = 0
        img2_masked[img2.shape[0] // 4:, :] = 0
    else:
        raise ValueError("Invalid mode. Use 'row' or 'col'.")
    return img1_masked, img2_masked


def validate_stitched_dimensions(stitched_width, stitched_height, mode):
    if mode == "row":
        height = 512
        width = 896

        if stitched_height != height or stitched_width != width:
            return False

    elif mode == "col":
        height = 896
        width = 512

        if stitched_height != height or stitched_width != width:
            return False

    else:
        raise ValueError("Invalid mode. Use 'row' for horizontal stitching or 'col' for vertical stitching.")

    return True


def compute_image_corners(img, homography=None):
    rows, cols = img.shape[:2]
    corners = np.float32([[0, 0], [0, rows], [cols, rows], [cols, 0]]).reshape(-1, 1, 2)
    if homography is not None:
        corners = cv2.perspectiveTransform(corners, homography)
    return corners


def monster_detector(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    step = 10
    h, w = gray1.shape

    def get_harris_keypoints(gray, quality_level=0.01):
        harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        threshold = quality_level * harris_corners.max()
        keypoints = [cv2.KeyPoint(x, y, step) for y in range(h) for x in range(w) if harris_corners[y, x] > threshold]
        return keypoints

    def get_shi_tomasi_keypoints(gray, maxCorners=500, qualityLevel=0.01, minDistance=10):
        corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
        if corners is not None:
            return [cv2.KeyPoint(c[0][0], c[0][1], 1) for c in corners]
        return []

    def get_dense_keypoints():
        return [cv2.KeyPoint(x, y, step) for y in range(0, h, step) for x in range(0, w, step)]

    fast = cv2.FastFeatureDetector_create()

    mser = cv2.MSER_create()

    def get_mser_keypoints(gray):
        regions, _ = mser.detectRegions(gray)
        return [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for region in regions for p in region]

    keypoints1 = []
    keypoints1.extend(get_harris_keypoints(gray1))
    keypoints1.extend(get_shi_tomasi_keypoints(gray1))
    keypoints1.extend(get_dense_keypoints())
    keypoints1.extend(fast.detect(gray1, None))
    keypoints1.extend(get_mser_keypoints(gray1))

    keypoints2 = []
    keypoints2.extend(get_harris_keypoints(gray2))
    keypoints2.extend(get_shi_tomasi_keypoints(gray2))
    keypoints2.extend(get_dense_keypoints())
    keypoints2.extend(fast.detect(gray2, None))
    keypoints2.extend(get_mser_keypoints(gray2))

    print(f"Keypoints detected: {len(keypoints1)} in image 1, {len(keypoints2)} in image 2")

    def remove_duplicates(keypoints):
        seen = set()
        unique_keypoints = []
        for kp in keypoints:
            coord = (int(kp.pt[0]), int(kp.pt[1]))
            if coord not in seen:
                seen.add(coord)
                unique_keypoints.append(kp)
        return unique_keypoints

    keypoints1 = remove_duplicates(keypoints1)
    keypoints2 = remove_duplicates(keypoints2)

    print(f"Unique keypoints: {len(keypoints1)} in image 1, {len(keypoints2)} in image 2")

    sift = cv2.SIFT_create(5000)
    kp1, des1 = sift.compute(gray1, keypoints1)
    kp2, des2 = sift.compute(gray2, keypoints2)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("Error: No descriptors found for matching.")
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Found {len(matches)} matches")

    return kp1, kp2, matches[:50]

def stitch_images_sift(img1, img2, mode):
    sift = cv2.SIFT_create(nfeatures=5000)

    img1_masked, img2_masked = mask_images(img1, img2, mode)

    try:
        kp1, des1 = sift.detectAndCompute(img1_masked, None)
        kp2, des2 = sift.detectAndCompute(img2_masked, None)

        if len(kp1) == 0 or len(kp2) == 0:
            raise ValueError("No keypoints found in the images.")

        matches = match_features(des1, des2, 'BF', cross_check=True)

        if len(kp1) > 3 and len(kp2) > 3:
            H, mask = compute_homography(kp1, kp2, matches)
            if H is None or not np.isfinite(H).all():
                raise ValueError("Initial homography matrix invalid.")
        else:
            raise ValueError("Initial homography matrix invalid.")

        points1 = compute_image_corners(img1)
        points2_transformed = compute_image_corners(img2, homography=H)
        all_points = np.concatenate((points1, points2_transformed), axis=0)

        [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

        stitched_width = x_max - x_min
        stitched_height = y_max - y_min

        if not validate_stitched_dimensions(stitched_width, stitched_height, mode):
            raise ValueError("Invalid dimensions with regular SIFT.")

    except ValueError as e:
        print(f"First attempt failed: {str(e)}")
        try:
            # Second attempt: monster detector
            kp1, kp2, matches = monster_detector(img1_masked, img2_masked)
            if not matches:
                raise ValueError("Dense SIFT matching failed.")

            if len(kp1) > 3 and len(kp2) > 3:
                H, mask = compute_homography(kp1, kp2, matches)
                if H is None or not np.isfinite(H).all():
                    raise ValueError("Dense SIFT homography matrix invalid.")
            else:
                raise ValueError("Dense SIFT homography matrix invalid.")

            points1 = compute_image_corners(img1)
            points2_transformed = compute_image_corners(img2, homography=H)
            all_points = np.concatenate((points1, points2_transformed), axis=0)

            [x_min, y_min] = np.int32(all_points.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_points.max(axis=0).ravel() + 0.5)

            stitched_width = x_max - x_min
            stitched_height = y_max - y_min

            if not validate_stitched_dimensions(stitched_width, stitched_height, mode):
                raise ValueError("Invalid dimensions with dense SIFT.")

        except ValueError as e:
            print(f"Second attempt failed: {str(e)}")
            print("Falling back to simple concatenation...")

            # Third attempt: Simple concatenation
            if mode == 'col':
                result = np.zeros((896, 512, 3), dtype=np.uint8)
                result[0:512, :] = img1
                result[512:, :] = img2[128:, :]
            else:  # row mode
                result = np.zeros((512, 896, 3), dtype=np.uint8)
                result[:, 0:512] = img1
                result[:, 512:] = img2[:, 128:]

            return result

    # If we get here, either first or second attempt succeeded
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]).dot(H)
    output_shape = (stitched_width, stitched_height)

    warped_img1 = cv2.warpPerspective(img1, H_translation, output_shape)
    warped_img1[-y_min:img2.shape[0] - y_min, -x_min:img2.shape[1] - x_min] = img2

    return warped_img1


def process_tiles_sift(input_dir, output_dir, mode='row'):
    os.makedirs(output_dir, exist_ok=True)

    tile_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith('.tif') or f.lower().endswith('.png')])
    tile_paths = [os.path.join(input_dir, f) for f in tile_files]

    processed_images = set()

    print(f"Loaded {len(tile_paths)} files from {input_dir}.")

    for i, tile_path in enumerate(tile_paths):
        print(f"Trying to load image: {tile_path}")

        if not os.path.exists(tile_path):
            print(f"File does not exist: {tile_path}")
            continue

        img1 = cv2.imread(tile_path)
        if img1 is None:
            print(f"Error reading image: {tile_path}")
            continue

        y_index, x_index = extract_row_col_filename(tile_files[i])

        if (y_index, x_index) in processed_images:
            print(f"Already processed: {tile_path}")
            continue

        processed_images.add((y_index, x_index))

        neighbors = find_neighbors_sub(y_index, x_index, tile_files, input_dir, mode=mode)

        for neighbor_path in neighbors:
            print(f"Trying to load neighbor image: {neighbor_path}")

            if not os.path.exists(neighbor_path):
                print(f"Neighbor file does not exist: {neighbor_path}")
                continue

            img2 = cv2.imread(neighbor_path)
            if img2 is None:
                print(f"Error reading neighbor image: {neighbor_path}")
                continue

            neighbor_y_index, neighbor_x_index = extract_row_col_filename(
                os.path.basename(neighbor_path))

            stitched_filename = f"stitched_{y_index}_{x_index}_{neighbor_y_index}_{neighbor_x_index}.tiff"
            stitched_path = os.path.join(output_dir, stitched_filename)

            if os.path.exists(stitched_path):
                print(f"Stitched image already exists: {stitched_path}")
                continue

            try:
                stitched = stitch_images_sift(img1, img2, mode)
                cv2.imwrite(stitched_path, stitched)
                print(f"Stitched and saved: {stitched_path}")

            except Exception as e:
                print(f"Error stitching {tile_path} and {neighbor_path}: {e}")

# process_tiles_sift('Philadelphia subtiles 512', 'stitched col 512', mode='col')

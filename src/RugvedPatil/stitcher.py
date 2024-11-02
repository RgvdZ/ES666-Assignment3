import cv2
import numpy as np
import glob
import logging
from scipy.optimize import least_squares

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanaromaStitcher:
    def __init__(self):
        # Use SIFT for feature detection
        self.sift = cv2.SIFT_create()

        # FLANN-based matcher for better performance
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)  # Specify how many times the tree should be traversed
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def make_panaroma_for_images_in(self, path):
        image_paths = sorted(glob.glob(f'{path}/*.*'))
        if len(image_paths) < 2:
            raise ValueError("Need at least two images to create a panorama")

        images = []
        for im_path in image_paths:
            img = cv2.imread(im_path)
            if img is None:
                raise ValueError(f"Error reading image: {im_path}")
            if 'I4' in path:
                img = self.resize_image(img, 800 / img.shape[1], 600 / img.shape[0])  # Custom resize
            images.append(img)

        # Initialize the panorama with the first image
        stitched_image = images[0]
        homography_matrix_list = []

        for i in range(1, len(images)):
            logger.info(f"Processing image {i+1}/{len(images)}")
            kp1, des1 = self.sift.detectAndCompute(stitched_image, None)
            kp2, des2 = self.sift.detectAndCompute(images[i], None)

            if des1 is None or des2 is None:
                logger.warning(f"No descriptors found in image {i}. Skipping this pair.")
                continue

            # Mutual (Cross-Check) Matching
            matches1to2 = self.matcher.knnMatch(des1, des2, k=2)
            matches2to1 = self.matcher.knnMatch(des2, des1, k=2)

            # Apply ratio test
            good_matches1to2 = set()
            for m, n in matches1to2:
                if m.distance < 0.75 * n.distance:
                    good_matches1to2.add(m.queryIdx)

            good_matches2to1 = set()
            for m, n in matches2to1:
                if m.distance < 0.75 * n.distance:
                    good_matches2to1.add(m.queryIdx)

            # Identify Mutual Matches
            mutual_matches = []
            for m in matches1to2:
              if m[0].queryIdx in good_matches1to2:
                m_rev_candidates = matches2to1[m[0].trainIdx]
                for m_rev in m_rev_candidates:
                  if m_rev.trainIdx == m[0].queryIdx:
                    mutual_matches.append(m[0])
                    break


            logger.info(f"Found {len(mutual_matches)} mutual good matches.")

            if len(mutual_matches) < 6:
                logger.warning(f"Not enough mutual good matches between image {i} and image {i-1}. Skipping this pair.")
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in mutual_matches]).reshape(-1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in mutual_matches]).reshape(-1, 2)

            # Compute homography with improved RANSAC
            H, inliers = self.compute_homography_ransac(pts1, pts2)
            if H is None:
                logger.warning(f"Failed to compute homography for image {i} and image {i-1}. Skipping this pair.")
                continue

            homography_matrix_list.append(H)
            # Refine homography with bundle adjustment using confidence weighting
            H_refined = self.bundle_adjustment(H, pts1[inliers], pts2[inliers])
            homography_matrix_list[-1] = H_refined

            # Warp and blend images
            stitched_image = self.blend_images(stitched_image, images[i], H_refined)

        return stitched_image, homography_matrix_list

    def normalize_points(self, pts):
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        avg_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))
        scale = np.sqrt(2) / avg_dist
        T = np.array([
            [scale, 0, -scale * mean[0]],
            [0, scale, -scale * mean[1]],
            [0,     0,               1]
        ])
        pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
        normalized_pts = (T @ pts_homogeneous.T).T
        return normalized_pts[:, :2], T

    def dlt(self, pts1, pts2):
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)
        A = []
        for i in range(len(pts1_norm)):
            x, y = pts1_norm[i]
            x_prime, y_prime = pts2_norm[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)
        try:
            U, S, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning None for homography.")
            return None
        H_norm = Vt[-1].reshape(3, 3)
        H = np.linalg.inv(T2) @ H_norm @ T1      # Denormalizing
        return H / H[2, 2]

    def compute_homography_ransac(self, pts1, pts2, max_iterations=2000, threshold=3.0, confidence=0.99):
        best_H = None
        max_inliers = 0
        best_inliers = []
        N = max_iterations
        for iteration in range(N):
            idx = np.random.choice(len(pts1), 4, replace=False)
            p1_sample = pts1[idx]
            p2_sample = pts2[idx]

            H_candidate = self.dlt(p1_sample, p2_sample)
            if H_candidate is None:
                continue

            # Project pts1 using H_candidate
            pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            projected_pts2_homogeneous = (H_candidate @ pts1_homogeneous.T).T
            # Avoid division by zero
            projected_pts2_homogeneous[:, 2] += 1e-10
            projected_pts2 = projected_pts2_homogeneous[:, :2] / projected_pts2_homogeneous[:, 2, np.newaxis]

            # Compute errors
            errors = np.linalg.norm(pts2 - projected_pts2, axis=1)
            inliers = np.where(errors < threshold)[0]
            num_inliers = len(inliers)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H_candidate
                best_inliers = inliers

                # Update the number of iterations based on inlier ratio
                inlier_ratio = num_inliers / len(pts1)
                eps = 1 - inlier_ratio
                if eps == 0:
                    eps = 1e-8  # Prevent division by zero
                N = min(N, int(np.log(1 - confidence) / np.log(1 - (1 - eps)**4)))
                if N < iteration + 1:
                    break

            # Early stopping if enough inliers are found
            if num_inliers > 0.8 * len(pts1):
                break

        if best_H is not None and len(best_inliers) >= 10:
            # Weighted least squares refinement
            best_H = self.weighted_least_squares(best_H, pts1[best_inliers], pts2[best_inliers])

            # Bundle adjustment with confidence weighting
            best_H = self.bundle_adjustment(best_H, pts1[best_inliers], pts2[best_inliers])
        else:
            logger.warning("Not enough inliers after RANSAC.")
            return None, None

        return best_H, best_inliers

    def weighted_least_squares(self, H, pts1, pts2):
        # Compute reprojection errors
        pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        projected_pts2_homogeneous = (H @ pts1_homogeneous.T).T
        projected_pts2 = projected_pts2_homogeneous[:, :2] / projected_pts2_homogeneous[:, 2, np.newaxis]
        errors = np.linalg.norm(pts2 - projected_pts2, axis=1)

        # Compute weights: inverse of error (add epsilon to avoid division by zero)
        epsilon = 1e-6
        weights = 1 / (errors + epsilon)
        weights /= np.sum(weights)

        # Construct weighted A matrix
        pts1_norm, T1 = self.normalize_points(pts1)
        pts2_norm, T2 = self.normalize_points(pts2)

        A = []
        for i in range(len(pts1_norm)):
            x, y = pts1_norm[i]
            x_prime, y_prime = pts2_norm[i]
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        A = np.array(A)

        # Repeat weights to match the number of rows in A
        weights_repeated = np.repeat(weights, 2)  # Shape (2N,)

        # Apply weights to A
        A_weighted = weights_repeated[:, np.newaxis] * A  # Shape (2N, 9)

        try:
            U, S, Vt = np.linalg.svd(A_weighted)
        except np.linalg.LinAlgError:
            logger.warning("Weighted SVD did not converge. Skipping refinement.")
            return H

        H_norm = Vt[-1].reshape(3, 3)
        H_refined = np.linalg.inv(T2) @ H_norm @ T1
        return H_refined / H_refined[2, 2]


    def bundle_adjustment(self, H_initial, pts1, pts2):
        # Compute confidence weights based on reprojection errors
        pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        projected_pts2_homogeneous = (H_initial @ pts1_homogeneous.T).T
        projected_pts2 = projected_pts2_homogeneous[:, :2] / projected_pts2_homogeneous[:, 2, np.newaxis]
        errors = np.linalg.norm(pts2 - projected_pts2, axis=1)

        # Compute confidence weights: higher weights for lower errors
        epsilon = 1e-6  # To prevent division by zero
        weights = 1 / (errors + epsilon)
        # Normalize weights to have maximum of 1
        weights /= np.max(weights)

        # Define the residuals function with confidence weighting
        def residuals(params, H_initial, pts1, pts2, weights):
            H = homography_matrix(params)
            pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            projected_pts2_homogeneous = (H @ pts1_homogeneous.T).T
            projected_pts2 = projected_pts2_homogeneous[:, :2] / projected_pts2_homogeneous[:, 2, np.newaxis]
            residual = (pts2 - projected_pts2).ravel()

            # Repeat weights to match residual shape
            weights_repeated = np.repeat(weights, 2)  # Shape (2N,)
            return residual * np.sqrt(weights_repeated)

        def homography_matrix(params):
            H = np.array([
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [0,         0,         1]
            ])
            return H

        # Initial parameters from H_initial
        params_initial = H_initial[:2, :].flatten()

        # Optimize using Levenberg-Marquardt with confidence weighting
        result = least_squares(
            residuals,
            params_initial,
            method='lm',
            args=(H_initial, pts1, pts2, weights)
        )

        if result.success:
            H_refined = homography_matrix(result.x)
            return H_refined / H_refined[2, 2]
        else:
            logger.warning("Bundle adjustment did not converge. Using initial homography.")
            return H_initial


    def blend_images(self, img1, img2, H):
        # Pre-filter img2 with Gaussian blur for anti-aliasing
        img2_filtered = cv2.GaussianBlur(img2, (3, 3), 0)

        # Warp img2 to img1's plane using multi-scale warping
        warped_img2 = self.manual_inverse_warp(img1, img2_filtered, H)

        print(f"Shape of img1: {img1.shape}")
        print(f"Shape of warped_img2: {warped_img2.shape}")

        # Determine the maximum dimensions
        max_height = max(img1.shape[0], warped_img2.shape[0])
        max_width = max(img1.shape[1], warped_img2.shape[1])

        # Calculate the next multiple of 16 for height and width
        height_padded = ((max_height + 15) // 16) * 16
        width_padded = ((max_width + 15) // 16) * 16

        # Pad images to have the same dimensions and be multiples of 16
        img1_padded = np.zeros((height_padded, width_padded, 3), dtype=img1.dtype)
        img1_padded[:img1.shape[0], :img1.shape[1]] = img1

        warped_img2_padded = np.zeros((height_padded, width_padded, 3), dtype=warped_img2.dtype)
        warped_img2_padded[:warped_img2.shape[0], :warped_img2.shape[1]] = warped_img2

        # Create masks
        mask1 = (cv2.cvtColor(img1_padded, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
        mask2 = (cv2.cvtColor(warped_img2_padded, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

        print(f"Shape of mask1: {mask1.shape}, dtype: {mask1.dtype}")
        print(f"Shape of mask2: {mask2.shape}, dtype: {mask2.dtype}")

        # Proceed with generating pyramids and blending
        blended_image = self.multi_band_blending(img1_padded, warped_img2_padded, mask1, mask2)

        print(f"Final blended image shape: {blended_image.shape}")

        return blended_image





    def manual_inverse_warp(self, img1, img2, H):
        # Pre-filter img2 with Gaussian blur for anti-aliasing
        img2_filtered = cv2.GaussianBlur(img2, (3, 3), 0)

        # Implement multi-scale warping using image pyramids
        num_levels = 3  # Number of pyramid levels
        H_pyramid = self.construct_pyramid(H, num_levels)

        stitched_image = img1.copy()
        for level in reversed(range(num_levels)):
            scale = 2 ** level
            H_level = H_pyramid[level]
            stitched_image = self.warp_image_pyramid(stitched_image, img2_filtered, H_level, scale)

        return stitched_image

    def construct_pyramid(self, H, levels):
        # Adjust homography for each pyramid level
        pyramid = []
        for level in range(levels):
            scale = 1 / (2 ** level)
            T = np.array([
                [scale,     0,    0],
                [0, scale,    0],
                [0,     0,    1]
            ])
            H_scaled = T @ H @ np.linalg.inv(T)
            pyramid.append(H_scaled)
        return pyramid

    def warp_image_pyramid(self, img1, img2, H, scale):
        # Resize images according to the current pyramid level using custom resize
        img1_scaled = self.resize_image(img1, scale, scale)
        img2_scaled = self.resize_image(img2, scale, scale)

        # Compute output size
        h1, w1 = img1_scaled.shape[:2]
        h2, w2 = img2_scaled.shape[:2]
        corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        transformed_corners = self.apply_homography_to_points(H, corners_img2)
        all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [w1, h1], [0, h1]]))
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        translation = np.array([[1, 0, -x_min],
                                [0, 1, -y_min],
                                [0, 0,        1]])

        H_translated = translation @ H

        output_shape = (y_max - y_min, x_max - x_min)
        warped_img2 = self.warp_image(img2_scaled, H_translated, output_shape)

        # Resize back to original scale using custom resize
        warped_img2 = self.resize_image(warped_img2, 1/scale, 1/scale)
        img1_scaled = self.resize_image(img1_scaled, 1/scale, 1/scale)

        # Initialize the output image
        stitched_image = np.zeros((output_shape[0], output_shape[1], 3), dtype=img1_scaled.dtype)
        stitched_image[-y_min:-y_min + img1_scaled.shape[0], -x_min:-x_min + img1_scaled.shape[1]] = img1_scaled

        # Masks
        mask1 = (stitched_image > 0).astype(np.float32)
        mask2 = (warped_img2 > 0).astype(np.float32)

        # Blend images using multi-band blending
        blended_image = self.multi_band_blending(stitched_image, warped_img2, mask1, mask2)

        return blended_image

    def apply_homography_to_points(self, H, pts):
        pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
        transformed_pts = (H @ pts_homogeneous.T).T
        transformed_pts /= transformed_pts[:, 2, np.newaxis]
        return transformed_pts[:, :2]

    def warp_image(self, img, H, output_shape):
        h_out, w_out = output_shape

        # Generate grid of (x, y) coordinates in the destination image
        xx, yy = np.meshgrid(np.arange(w_out), np.arange(h_out))
        ones = np.ones_like(xx)
        coords = np.stack([xx.ravel(), yy.ravel(), ones.ravel()], axis=1)  # Shape (N, 3)

        # Apply inverse homography to map destination pixels back to source image
        H_inv = np.linalg.inv(H)
        coords_transformed = coords @ H_inv.T
        coords_transformed /= coords_transformed[:, 2:3]  # Normalize homogeneous coordinates

        x_src = coords_transformed[:, 0]
        y_src = coords_transformed[:, 1]

        # Find valid coordinates within the bounds of the source image
        valid_indices = (
            (x_src >= 0) & (x_src < img.shape[1] - 1) &
            (y_src >= 0) & (y_src < img.shape[0] - 1)
        )

        # Corresponding destination coordinates
        x_dst = coords[valid_indices, 0].astype(np.int32)
        y_dst = coords[valid_indices, 1].astype(np.int32)

        # Source coordinates for interpolation
        x_src_valid = x_src[valid_indices]
        y_src_valid = y_src[valid_indices]

        # Perform bicubic interpolation
        warped_pixels = self.bicubic_interpolate(img, x_src_valid, y_src_valid)

        # Clip pixel values to valid range and cast to appropriate type
        warped_pixels = np.clip(warped_pixels, 0, 255).astype(img.dtype)

        # Initialize the output image
        warped_image = np.zeros((h_out, w_out, img.shape[2]), dtype=img.dtype)

        # Assign warped pixels to the valid positions in the output image
        warped_image[y_dst, x_dst] = warped_pixels

        return warped_image

    def bicubic_interpolate(self, img, x, y):
        h, w = img.shape[:2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        channels = img.shape[2]

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        dx = x - x0
        dy = y - y0

        m = np.array([-1, 0, 1, 2])
        n = np.array([-1, 0, 1, 2])

        dx_diff = dx[:, np.newaxis] - m[np.newaxis, :]  # Shape (num_points, 4)
        dy_diff = dy[:, np.newaxis] - n[np.newaxis, :]  # Shape (num_points, 4)

        wx = self.cubic(dx_diff)  # Shape (num_points, 4)
        wy = self.cubic(dy_diff)  # Shape (num_points, 4)

        # Compute weights for 4x4 neighborhood
        weights = wy[:, :, np.newaxis] * wx[:, np.newaxis, :]  # Shape (num_points, 4, 4)

        # Get neighbor indices
        xi = x0[:, np.newaxis] + m[np.newaxis, :]  # Shape (num_points, 4)
        yi = y0[:, np.newaxis] + n[np.newaxis, :]  # Shape (num_points, 4)

        # Clip indices to valid range
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)

        # Expand yi and xi to (num_points, 4, 4)
        yi_exp = np.repeat(yi[:, :, np.newaxis], 4, axis=2)  # Shape (num_points, 4, 4)
        xi_exp = np.repeat(xi[:, np.newaxis, :], 4, axis=1)  # Shape (num_points, 4, 4)

        # Prepare output array
        output = np.zeros((x.size, channels))

        for c in range(channels):
            pixels = img[yi_exp, xi_exp, c]  # Shape (num_points, 4, 4)
            weighted_pixels = pixels * weights  # Shape (num_points, 4, 4)
            output[:, c] = np.sum(weighted_pixels, axis=(1, 2))

        return output

    def cubic(self, x, a=-0.5):
        absx = np.abs(x)
        absx2 = absx ** 2
        absx3 = absx2 * absx

        h = np.zeros_like(x)

        mask1 = (absx <= 1)
        mask2 = (absx > 1) & (absx < 2)

        h[mask1] = 1.5 * absx3[mask1] - 2.5 * absx2[mask1] + 1
        h[mask2] = -0.5 * absx3[mask2] + 2.5 * absx2[mask2] - 4 * absx[mask2] + 2

        return h

    def multi_band_blending(self, img1, img2, mask1, mask2):
        levels = 5  # Number of pyramid levels

        # Predefine sizes for each level based on the padded image dimensions
        sizes = []
        h, w = img1.shape[:2]  # img1 and img2 are padded to the same size
        sizes.append((h, w))
        for i in range(1, levels):
            h = (h + 1) // 2
            w = (w + 1) // 2
            sizes.append((h, w))

        # Generate Gaussian pyramids for masks using predefined sizes
        gp_mask1 = self.generate_gaussian_pyramid_with_sizes(mask1, sizes, is_mask=True)
        gp_mask2 = self.generate_gaussian_pyramid_with_sizes(mask2, sizes, is_mask=True)

        # Generate Laplacian pyramids for images using predefined sizes
        lp_img1 = self.generate_laplacian_pyramid_with_sizes(img1, sizes)
        lp_img2 = self.generate_laplacian_pyramid_with_sizes(img2, sizes)

        # Blend pyramids
        blended_pyramid = []
        for i in range(levels):
            print(f"Level {i}:")
            print(f"  gp_mask1[{i}].shape: {gp_mask1[i].shape}, dtype: {gp_mask1[i].dtype}, ndim: {gp_mask1[i].ndim}")
            print(f"  gp_mask2[{i}].shape: {gp_mask2[i].shape}, dtype: {gp_mask2[i].dtype}, ndim: {gp_mask2[i].ndim}")
            print(f"  lp_img1[{i}].shape: {lp_img1[i].shape}, dtype: {lp_img1[i].dtype}")
            print(f"  lp_img2[{i}].shape: {lp_img2[i].shape}, dtype: {lp_img2[i].dtype}")

            # Ensure masks are 2D arrays
            if gp_mask1[i].ndim > 2:
                gp_mask1[i] = gp_mask1[i][:, :, 0]
            if gp_mask2[i].ndim > 2:
                gp_mask2[i] = gp_mask2[i][:, :, 0]

            # Expand masks to match the number of channels
            gp_mask1_expanded = gp_mask1[i][:, :, np.newaxis]
            gp_mask2_expanded = gp_mask2[i][:, :, np.newaxis]

            # Ensure masks have the same number of channels as images
            gp_mask1_expanded = np.repeat(gp_mask1_expanded, lp_img1[i].shape[2], axis=2)
            gp_mask2_expanded = np.repeat(gp_mask2_expanded, lp_img2[i].shape[2], axis=2)

            # Blend images
            blended = gp_mask1_expanded * lp_img1[i] + gp_mask2_expanded * lp_img2[i]
            blended_pyramid.append(blended)

        # Reconstruct image from blended pyramid
        blended_image = self.reconstruct_from_pyramid(blended_pyramid)

        # Clip values to valid range
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

        return blended_image



    def generate_gaussian_pyramid(self, img, levels, is_mask=False):
        sizes = []
        h, w = img.shape[:2]
        sizes.append((h, w))
        for i in range(1, levels):
            h = (h + 1) // 2  # Handle odd sizes
            w = (w + 1) // 2
            sizes.append((h, w))

        gaussian_pyramid = []
        current_img = img.copy()
        for idx, size in enumerate(sizes):
            if is_mask:
                img_resized = cv2.resize(current_img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                img_resized = cv2.resize(current_img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

            # Ensure that the resized image has the same number of dimensions as the input
            if current_img.ndim == 2 and img_resized.ndim == 3:
                img_resized = img_resized[:, :, 0]
            gaussian_pyramid.append(img_resized)
            current_img = img_resized
        return gaussian_pyramid


    def generate_gaussian_pyramid_with_sizes(self, img, sizes, is_mask=False):
        gaussian_pyramid = []
        current_img = img.copy()
        for idx, size in enumerate(sizes):
            if is_mask:
                # Ensure the image is 2D
                if current_img.ndim == 3:
                    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(current_img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
                # Ensure the resized image is 2D
                if img_resized.ndim == 3:
                    img_resized = img_resized[:, :, 0]
            else:
                img_resized = cv2.resize(current_img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            gaussian_pyramid.append(img_resized)
            current_img = img_resized
        return gaussian_pyramid



    def generate_laplacian_pyramid_with_sizes(self, img, sizes):
        gaussian_pyramid = self.generate_gaussian_pyramid_with_sizes(img, sizes)
        laplacian_pyramid = []
        for i in range(len(sizes) - 1):
            size = sizes[i]
            gaussian_expanded = cv2.resize(gaussian_pyramid[i + 1], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid





    def reconstruct_from_pyramid(self, pyramid):
        reconstructed = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            reconstructed = cv2.resize(reconstructed, size, interpolation=cv2.INTER_LINEAR)
            reconstructed = cv2.add(reconstructed, pyramid[i])
        return reconstructed




    def resize_image(self, img, fx, fy, output_shape=None):
        h, w = img.shape[:2]

        if output_shape is not None:
            new_h, new_w = output_shape
        else:
            new_w = int(w * fx)
            new_h = int(h * fy)

        if new_w <= 0 or new_h <= 0:
            raise ValueError("Scale factors result in non-positive dimensions.")

        # Generate grid for the resized image
        x_new = np.linspace(0, w - 1, new_w)
        y_new = np.linspace(0, h - 1, new_h)
        xx, yy = np.meshgrid(x_new, y_new)

        xx_flat = xx.ravel()
        yy_flat = yy.ravel()

        # Perform bicubic interpolation
        interpolated_pixels = self.bicubic_interpolate(img, xx_flat, yy_flat)

        # Reshape back to image format
        if interpolated_pixels.shape[1] == 1:
            interpolated_image = interpolated_pixels.reshape((new_h, new_w))
        else:
            interpolated_image = interpolated_pixels.reshape((new_h, new_w, -1))

        return interpolated_image

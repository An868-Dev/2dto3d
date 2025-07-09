import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

def apply_mask(image_path, mask_path, crop=True):
    # Load ảnh RGB và mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[⚠] Lỗi đọc file ảnh: {image_path}")
        return None
    if mask is None:
        print(f"[⚠] Lỗi đọc file mask: {mask_path}")
        return None

    # Kiểm tra và thay đổi kích thước mask nếu cần
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Chuẩn hóa mask: nhị phân (0 hoặc 255)
    mask_bin = (mask > 128).astype(np.uint8) * 255

    # Apply mask để giữ foreground
    masked = cv2.bitwise_and(image, image, mask=mask_bin)

    # Tùy chọn: crop quanh vật thể
    if crop:
        coords = cv2.findNonZero(mask_bin)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            masked = masked[y:y+h, x:x+w]

    return masked


def process_dataset(images_dir, masks_dir, output_dir, crop=True):
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(glob.glob(os.path.join(images_dir, '**', ext), recursive=True)))

    total = len(image_files)

    print(f"[ℹ] Đang xử lý {total} ảnh...")
    for image_path in tqdm(image_files):
        # Lấy tên file không có extension
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        # Tạo đường dẫn cho mask và output
        relative_path = os.path.relpath(os.path.dirname(image_path), images_dir)
        mask_filename = base_filename + ".png" # Mask luôn là file .png
        mask_path = os.path.join(masks_dir, relative_path, mask_filename)
        output_filename = base_filename + ".png" # Lưu kết quả dưới dạng .png
        output_path = os.path.join(output_dir, relative_path, output_filename)

        # Kiểm tra sự tồn tại của mask
        if not os.path.exists(mask_path):
            print(f"[⚠] Không tìm thấy mask cho ảnh: {image_path}. Bỏ qua.")
            continue

        # Đảm bảo thư mục output tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result = apply_mask(image_path, mask_path, crop=crop)
        if result is not None:
            cv2.imwrite(output_path, result)


if __name__ == "__main__":
    # Đường dẫn đến thư mục dữ liệu
    images_dir = "ytre/data/pix3d/images"           # chứa ảnh RGB gốc
    masks_dir = "ytre/data/pix3d/masks"             # chứa ảnh mask nhị phân
    output_dir = "ytre/data/pix3d/masked_images"    # nơi lưu kết quả

    # Gọi hàm chính
    process_dataset(images_dir, masks_dir, output_dir, crop=True)

    print("[✅] Đã xử lý xong tất cả ảnh và lưu vào thư mục:", output_dir)
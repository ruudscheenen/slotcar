# pyright: reportOptionalMemberAccess=false
from ctypes import c_ubyte

import cv2
import numpy as np
from _ctypes import addressof
from gxipy import GxPixelFormatEntry, DxValidBit
import gxipy as gx

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

# Prepare object points
# The example chessboard is 9x6.
CHESSBOARD_X = 7
CHESSBOARD_Y = 7
# The example chessboard printed on a A4 paper will approximately be 22.5mm.
SQUARE_SIZE_MM = 10

objp = np.zeros((CHESSBOARD_X * CHESSBOARD_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_X, 0:CHESSBOARD_Y].T.reshape(-1, 2) * (SQUARE_SIZE_MM * 0.001)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

img_counter = 0

# ===== BRIGHTNESS CONTROLS - ADJUST THESE =====
EXPOSURE_TIME = 100000.0  # Microseconds (10ms). Increase to brighten, decrease to darken
GAIN = 0.0            # dB. Range typically 0-20. Higher = brighter but more noise
# ==============================================

def get_best_valid_bits(pixel_format):
    """Determine the best valid bits based on pixel format."""
    valid_bits = DxValidBit.BIT0_7
    if pixel_format in (GxPixelFormatEntry.MONO8,
                        GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8,
                        GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8,
                        GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8,
                        GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
        valid_bits = DxValidBit.BIT0_7
    elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.MONO10_P,
                          GxPixelFormatEntry.BAYER_GR10, GxPixelFormatEntry.BAYER_RG10,
                          GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10,
                          GxPixelFormatEntry.BAYER_GR10_P, GxPixelFormatEntry.BAYER_RG10_P,
                          GxPixelFormatEntry.BAYER_GB10_P, GxPixelFormatEntry.BAYER_BG10_P,
                          GxPixelFormatEntry.BAYER_GR10_PACKED, GxPixelFormatEntry.BAYER_RG10_PACKED,
                          GxPixelFormatEntry.BAYER_GB10_PACKED, GxPixelFormatEntry.BAYER_BG10_PACKED):
        valid_bits = DxValidBit.BIT2_9
    elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.MONO12_P,
                          GxPixelFormatEntry.BAYER_GR12, GxPixelFormatEntry.BAYER_RG12,
                          GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12,
                          GxPixelFormatEntry.BAYER_GR12_P, GxPixelFormatEntry.BAYER_RG12_P,
                          GxPixelFormatEntry.BAYER_GB12_P, GxPixelFormatEntry.BAYER_BG12_P,
                          GxPixelFormatEntry.BAYER_GR12_PACKED, GxPixelFormatEntry.BAYER_RG12_PACKED,
                          GxPixelFormatEntry.BAYER_GB12_PACKED, GxPixelFormatEntry.BAYER_BG12_PACKED):
        valid_bits = DxValidBit.BIT4_11
    elif pixel_format in (GxPixelFormatEntry.MONO14, GxPixelFormatEntry.MONO14_P,
                          GxPixelFormatEntry.BAYER_GR14, GxPixelFormatEntry.BAYER_RG14,
                          GxPixelFormatEntry.BAYER_GB14, GxPixelFormatEntry.BAYER_BG14,
                          GxPixelFormatEntry.BAYER_GR14_P, GxPixelFormatEntry.BAYER_RG14_P,
                          GxPixelFormatEntry.BAYER_GB14_P, GxPixelFormatEntry.BAYER_BG14_P):
        valid_bits = DxValidBit.BIT6_13
    elif pixel_format in (GxPixelFormatEntry.MONO16,
                          GxPixelFormatEntry.BAYER_GR16, GxPixelFormatEntry.BAYER_RG16,
                          GxPixelFormatEntry.BAYER_GB16, GxPixelFormatEntry.BAYER_BG16):
        valid_bits = DxValidBit.BIT8_15
    return valid_bits


def convert_to_rgb(image_convert, raw_image):
    """Convert raw camera image to RGB using the library's converter."""
    image_convert.set_dest_format(GxPixelFormatEntry.RGB8)
    valid_bits = get_best_valid_bits(raw_image.get_pixel_format())
    image_convert.set_valid_bits(valid_bits)

    # Create output buffer
    buffer_out_size = image_convert.get_buffer_size_for_conversion(raw_image)
    output_image_array = (c_ubyte * buffer_out_size)()
    output_image = addressof(output_image_array)

    # Convert to RGB
    image_convert.convert(raw_image, output_image, buffer_out_size, False)
    if output_image is None:
        print('Failed to convert RawImage to RGB')
        return None, None

    return output_image_array, buffer_out_size


def main():
    # Create device manager
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()

    if dev_num == 0:
        print("No camera found")
        return

    # Open the camera
    cam = device_manager.open_device_by_index(1)

    # Get image converter
    image_convert = device_manager.create_image_format_convert()

    # Get remote device features to check pixel format
    remote_device_feature = cam.get_remote_device_feature_control()
    pixel_format_value, pixel_format_str = remote_device_feature.get_enum_feature("PixelFormat").get()
    print(f"Camera pixel format: {pixel_format_str}")

    # Set continuous acquisition
    trigger_mode_feature = remote_device_feature.get_enum_feature("TriggerMode")
    trigger_mode_feature.set("Off")

    # Set exposure mode
    exposure_mode = remote_device_feature.get_enum_feature("ExposureAuto")
    exposure_mode.set("Off")

    # Set gain mode
    gain_mode = remote_device_feature.get_enum_feature("GainAuto")
    gain_mode.set("Off")

    # Configure brightness settings
    print(f"\nConfiguring brightness:")

    # Set exposure time (use int feature)
    try:
        if remote_device_feature.is_implemented("ExposureTime"):
            exposure_range = remote_device_feature.get_float_feature("ExposureTime").get_range()
            print(f"  Exposure range: {exposure_range['min']} - {exposure_range['max']} μs")
            exposure_to_set = max(exposure_range['min'], min(EXPOSURE_TIME, exposure_range['max']))
            #exposure_to_set = EXPOSURE_TIME
            remote_device_feature.get_float_feature("ExposureTime").set(exposure_to_set)
            print(f"  Exposure set to: {exposure_to_set} μs")
        else:
            print("  ExposureTime not available")
    except Exception as e:
        print(f"  Failed to set exposure: {e}")

    # Set gain (try float first, fallback to int)
    try:
        if remote_device_feature.is_implemented("Gain"):
            try:
                gain_range = remote_device_feature.get_float_feature("Gain").get_range()
                print(f"  Gain range: {gain_range['min']:.1f} - {gain_range['max']:.1f} dB")
                gain_to_set = max(gain_range['min'], min(GAIN, gain_range['max']))
                remote_device_feature.get_float_feature("Gain").set(gain_to_set)
                print(f"  Gain 1 set to: {gain_to_set:.1f} dB")
            except:
                gain_range = remote_device_feature.get_int_feature("Gain").get_range()
                print(f"  Gain range: {gain_range['min']} - {gain_range['max']} dB")
                gain_to_set = int(max(gain_range['min'], min(GAIN, gain_range['max'])))
                remote_device_feature.get_int_feature("Gain").set(gain_to_set)
                print(f"  Gain set to: {gain_to_set} dB")
        else:
            print("  Gain not available")
    except Exception as e:
        print(f"  Failed to set gain: {e}")

    print()

    # Start streaming
    cam.stream_on()

    print("Press 'q' to quit")

    img_counter = 0
    while True:

        # Get raw image
        raw_image = cam.data_stream[0].get_image()

        if raw_image is None:
            continue

        # Convert based on pixel format
        if raw_image.get_pixel_format() == GxPixelFormatEntry.RGB8:
            # Already RGB
            numpy_image = raw_image.get_numpy_array()
            # bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        else:
            # Need to convert (likely Bayer pattern)
            rgb_image_array, rgb_buffer_length = convert_to_rgb(image_convert, raw_image)

            if rgb_image_array is None:
                continue

            # Create numpy array from converted RGB data
            numpy_image = np.frombuffer(rgb_image_array, dtype=np.ubyte, count=rgb_buffer_length).reshape(
                raw_image.frame_data.height, raw_image.frame_data.width, 3
            )

            # Convert RGB to BGR for OpenCV
            # bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # frame_clean = cv2.copyTo(frame, None)
        # Find the chess board corners
        gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X, CHESSBOARD_Y), flags)

        if ret:
            cv2.drawChessboardCorners(numpy_image, (CHESSBOARD_Y, CHESSBOARD_X), corners, ret)

        cv2.imshow("Test", numpy_image)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, numpy_image)
            print(f"{img_name} written!")
            img_counter += 1

            # Convert to grayscale
            # gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X, CHESSBOARD_Y), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(corners2)
            else:
                print("No chessboard detected in this image: {img_name}")


    if len(objpoints) > 0:
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Print out the camera calibration results
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)

        np.savez(
            "camera_calibration.npz",
            camera_matrix=mtx,
            dist_coeffs=dist,
            rvecs=rvecs,
            tvecs=tvecs
        )

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
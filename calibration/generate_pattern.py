"""Tool to generate printable checkerboard calibration patterns."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def generate_checkerboard_pattern(
    grid_cols: int = 9,
    grid_rows: int = 7,
    square_size_px: int = 150,
    margin_px: int = 120,
    output_path: Union[str, Path] = "calibration/checkerboard_pattern.png",
) -> Path:
    """Generates a high-contrast printable checkerboard image.

    Note:
        Number of squares horizontally = grid_cols + 1
        Number of squares vertically   = grid_rows + 1
        This guarantees exactly grid_cols x grid_rows internal corner vertices.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    n_squares_x = grid_cols + 1
    n_squares_y = grid_rows + 1

    board_w = n_squares_x * square_size_px
    board_h = n_squares_y * square_size_px

    img_w = board_w + 2 * margin_px
    img_h = board_h + 2 * margin_px + 80  # Extra space for scale bar and labels

    # White background
    image = np.ones((img_h, img_w), dtype=np.uint8) * 255

    # Draw squares
    for y in range(n_squares_y):
        for x in range(n_squares_x):
            if (x + y) % 2 == 1:
                top_left_x = margin_px + x * square_size_px
                top_left_y = margin_px + y * square_size_px
                cv2.rectangle(
                    image,
                    (top_left_x, top_left_y),
                    (top_left_x + square_size_px, top_left_y + square_size_px),
                    0,
                    -1,
                )

    # Annotate specs and 100mm verification scale bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_spec = f"Checkerboard Grid: {grid_cols}x{grid_rows} Inner Corners ({n_squares_x}x{n_squares_y} Squares)"
    text_inst = "CRITICAL: Measure a square with your calliper before calibration; do not assume printed scale."

    cv2.putText(image, text_spec, (margin_px, margin_px // 2), font, 0.7, 0, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        text_inst,
        (margin_px, board_h + margin_px + 50),
        font,
        0.55,
        0,
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(target_path), image)
    return target_path


if __name__ == "__main__":
    out = generate_checkerboard_pattern()
    print(f"Generated printable checkerboard pattern: {out}")

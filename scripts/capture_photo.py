from __future__ import annotations

from _bootstrap import add_repo_root
add_repo_root()


from camera_tools import DEFAULT_CAMERA_INDEX, capture_photo


def main() -> None:
    """Capture a photo and save it to disk."""
    try:
        output_path = capture_photo(camera_index=DEFAULT_CAMERA_INDEX)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    print(f"Saved photo to {output_path.resolve()}")


if __name__ == "__main__":
    main()

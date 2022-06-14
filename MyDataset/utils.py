from pathlib import Path


def get_files(path: Path) -> list:

    files = []
    for dir in path.iterdir():
        for img in dir.iterdir():
            files.append(img)
    return files


def get_stacks(images_path: Path, masks_path: Path, tubules: list) -> list:
    """
    Returns: list of ((img1, img2, img3), path_mask, dest_name)

    For example:
    [
        (
            (
                PosixPath('tubule 1 zstack 1.jpg'),
                PosixPath('tubule 1 zstack 2.jpg'),
                PosixPath('tubule 1 zstack 3.jpg'),
            ),
            PosixPath('tubule 1 zstack 2.png'),
            'tubule1zstack1__tubule1zstack2__tubule1zstack3'
        )
    ]
    """
    stacks = []
    # e.g. Sett 2019/tubule 1
    for dir_tubule in tubules:
        dir_tubule = images_path / dir_tubule
        # e.g. tubule 1 z-stack 3
        for dir_type in dir_tubule.iterdir():
            images = [img for img in dir_type.iterdir()]
            images.sort()
            # Stack without first image
            paths_img = ("", images[0], images[1])
            stack = _get_stack(paths_img, masks_path)
            # If stack mask does not exist skip
            if stack:
                stacks.append(stack)
            for i in range(1, len(images) - 1):
                # Get images
                paths_img = (images[i - 1], images[i], images[i + 1])
                stack = _get_stack(paths_img, masks_path)
                if stack:
                    stacks.append(stack)
            # Stack without last image
            paths_img = (images[-2], images[-1], "")
            stack = _get_stack(paths_img, masks_path)
            if stack:
                stacks.append(stack)
    return stacks


def _get_stack(paths_img, masks_path):
    img_prev, img_mid, img_next = paths_img
    mask = masks_path / f"{img_mid.stem}.png"
    if not mask.exists():
        # print(f"Mask does not exist, skipping: {mask.name}")
        return
    img_prev = img_prev.stem if img_prev else "empty"
    img_mid = img_mid.stem
    img_next = img_next.stem if img_next else "empty"
    name = (
        img_prev.replace(" ", "")
        + "__"
        + img_mid.replace(" ", "")
        + "__"
        + img_next.replace(" ", "")
    )
    return paths_img, mask, img_mid # paths_img, mask, name


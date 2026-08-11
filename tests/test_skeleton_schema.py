from src.data.skeleton_schema import (
    NUM_JOINTS,
    hip_joints,
    joint_ids,
    joint_names,
    lower_limb_joints,
    root_joint,
    shoulder_joints,
    skeleton_edges,
)


def test_coco17_schema_and_edges_are_valid() -> None:
    assert NUM_JOINTS == 17
    assert len(joint_names) == len(joint_ids) == 17
    assert root_joint == "pelvis_midpoint"
    assert hip_joints == (joint_ids["left_hip"], joint_ids["right_hip"])
    assert len(lower_limb_joints) == 6
    assert len(shoulder_joints) == 2
    assert skeleton_edges
    for left, right in skeleton_edges:
        assert 0 <= left < 17
        assert 0 <= right < 17
        assert left != right

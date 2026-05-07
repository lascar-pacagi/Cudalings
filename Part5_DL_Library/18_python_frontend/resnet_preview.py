#!/usr/bin/env python3
"""
resnet_preview.py -- Preview of the ResNet architecture using cudalearn.

This script defines a ResNet (Residual Network) entirely in Python using
our cudalearn CUDA library. It mirrors the architecture that will be
implemented in C++ in the next chapter (Chapter 19: cnn_resnet).

ResNet Key Idea:
    Instead of learning H(x) directly, learn the residual F(x) = H(x) - x.
    The output is x + F(x), where F(x) is computed by a small sub-network.
    This "skip connection" helps gradients flow through deep networks.

Architecture Overview:
    Stem:       Conv2d(C_in, 16, 3, padding=1) -> BN -> ReLU
    Stage 1:    N ResBlocks with 16 filters
    Stage 2:    N ResBlocks with 32 filters (first block strides by 2)
    Stage 3:    N ResBlocks with 64 filters (first block strides by 2)
    Head:       BN -> ReLU -> GlobalAvgPool2d -> Linear(64, num_classes)

Each ResBlock:
    x --> Conv -> BN -> ReLU -> Conv -> BN --> (+) --> ReLU --> out
    |                                          ^
    +------------------------------------------+  (skip/identity connection)

This file demonstrates:
    - How to compose cudalearn layers into complex architectures
    - The ResBlock pattern with skip connections
    - Parameter counting for model summary
    - Forward pass with dummy data

The PyTorch equivalent for comparison:
    - Uses nn.Module subclassing and self.fc = nn.Linear(...)
    - Our cudalearn version uses Sequential containers and manual add()

Build the extension first:
    make                              (using the Makefile)
    python setup.py build_ext --inplace  (using setuptools)
"""

import sys
import os
import numpy as np

# =============================================================================
# Import cudalearn with helpful error message
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import cudalearn
except ImportError as e:
    print("=" * 60)
    print("ERROR: Could not import 'cudalearn'.")
    print()
    print("The C++ extension module has not been built yet.")
    print("Build it first with one of these commands:")
    print()
    print("  cd", script_dir)
    print("  make                                # using Makefile")
    print("  python setup.py build_ext --inplace # using setuptools")
    print()
    print("Original error:", e)
    print("=" * 60)
    sys.exit(1)


# =============================================================================
# ResBlock: A single residual block
# =============================================================================
# A ResBlock computes:
#     out = ReLU( x + F(x) )
# where F(x) = BN(Conv(ReLU(BN(Conv(x)))))
#
# If the input and output have different numbers of channels (or different
# spatial dimensions due to stride > 1), we add a 1x1 "projection" convolution
# on the skip path to match dimensions:
#     out = ReLU( proj(x) + F(x) )
#
# PyTorch equivalent:
#     class ResBlock(nn.Module):
#         def __init__(self, in_ch, out_ch, stride=1):
#             super().__init__()
#             self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
#             self.bn1 = nn.BatchNorm2d(out_ch)
#             self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
#             self.bn2 = nn.BatchNorm2d(out_ch)
#             self.relu = nn.ReLU()
#             if stride != 1 or in_ch != out_ch:
#                 self.downsample = nn.Sequential(
#                     nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
#                     nn.BatchNorm2d(out_ch))

class ResBlock:
    """
    A residual block with two 3x3 convolutions and a skip connection.

    Architecture:
        x --+--> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -->(+)--> ReLU --> out
            |                                               ^
            +--- [optional 1x1 projection] ----------------+

    The projection is used when in_channels != out_channels or stride != 1,
    to match the spatial dimensions and channel count of the skip path.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create a ResBlock.

        Args:
            in_channels:  Number of input channels
            out_channels: Number of output channels
            stride:       Stride for the first convolution (1 = same size,
                          2 = downsample spatial dimensions by 2x)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # ---- Main path: two 3x3 convolutions ----
        # The first conv may downsample (stride > 1)
        # We use use_bias=False because BatchNorm has its own bias (beta)

        # Sequential container for the main path F(x)
        # PyTorch: self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.main_path = cudalearn.Sequential()

        # Conv 1: may change channels and/or downsample
        self.main_path.add("conv1",
            cudalearn.Conv2d(in_channels, out_channels, 3,
                             padding=1, stride=stride, use_bias=False))

        # BatchNorm after first conv
        # PyTorch: self.bn1 = nn.BatchNorm2d(out_ch)
        self.main_path.add("bn1", cudalearn.BatchNorm2d(out_channels))

        # ReLU activation
        # PyTorch: self.relu = nn.ReLU(inplace=True)
        self.main_path.add("relu1", cudalearn.ReLU())

        # Conv 2: preserves channels and spatial size (stride=1, padding=1)
        # PyTorch: self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.main_path.add("conv2",
            cudalearn.Conv2d(out_channels, out_channels, 3,
                             padding=1, stride=1, use_bias=False))

        # BatchNorm after second conv (ReLU comes AFTER the residual addition)
        # PyTorch: self.bn2 = nn.BatchNorm2d(out_ch)
        self.main_path.add("bn2", cudalearn.BatchNorm2d(out_channels))

        # ---- Skip/shortcut path ----
        # If dimensions change, we need a 1x1 projection convolution
        # to match the main path output shape.
        #
        # When stride=1 and in_channels==out_channels:
        #   skip(x) = x  (identity, no extra parameters)
        #
        # When stride>1 or in_channels!=out_channels:
        #   skip(x) = BN(Conv1x1(x, stride))  (learned projection)
        self.has_projection = (stride != 1 or in_channels != out_channels)

        if self.has_projection:
            # PyTorch: self.downsample = nn.Sequential(
            #     nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            #     nn.BatchNorm2d(out_ch))
            self.skip_path = cudalearn.Sequential()
            self.skip_path.add("proj_conv",
                cudalearn.Conv2d(in_channels, out_channels, 1,
                                 padding=0, stride=stride, use_bias=False))
            self.skip_path.add("proj_bn",
                cudalearn.BatchNorm2d(out_channels))
        else:
            self.skip_path = None

        # ReLU applied after the residual addition
        # Note: in our simplified implementation, we apply ReLU as a separate
        # step after manually adding the skip connection.
        self.final_relu = cudalearn.ReLU()

    def parameters(self):
        """Collect all learnable parameters from both paths."""
        params = list(self.main_path.parameters())
        if self.skip_path is not None:
            params += list(self.skip_path.parameters())
        # final_relu has no parameters
        return params

    def num_params(self):
        """Count total learnable parameters in this block."""
        return sum(p.size for p in self.parameters())

    def summary_str(self):
        """Human-readable summary of this block."""
        proj = " + 1x1 projection" if self.has_projection else ""
        return (f"ResBlock({self.in_channels} -> {self.out_channels}, "
                f"stride={self.stride}{proj})")


# =============================================================================
# ResNet: Full residual network
# =============================================================================
# Architecture (for CIFAR-style small images):
#
#   Stem:
#       Conv2d(in_ch, 16, 3, padding=1) -> BN(16) -> ReLU
#       Produces 16-channel feature maps at original spatial resolution.
#
#   Stage 1: N ResBlocks with 16 channels (no downsampling)
#   Stage 2: N ResBlocks with 32 channels (first block strides by 2)
#   Stage 3: N ResBlocks with 64 channels (first block strides by 2)
#
#   Head:
#       GlobalAvgPool2d -> Linear(64, num_classes)
#       Reduces each 64-channel feature map to a single vector, then classifies.
#
# PyTorch equivalent:
#     class ResNet(nn.Module):
#         def __init__(self, num_blocks, in_channels, num_classes):
#             super().__init__()
#             self.stem = nn.Sequential(
#                 nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False),
#                 nn.BatchNorm2d(16), nn.ReLU())
#             self.stage1 = self._make_stage(16, 16, num_blocks, stride=1)
#             self.stage2 = self._make_stage(16, 32, num_blocks, stride=2)
#             self.stage3 = self._make_stage(32, 64, num_blocks, stride=2)
#             self.gap = nn.AdaptiveAvgPool2d(1)
#             self.fc = nn.Linear(64, num_classes)

class ResNet:
    """
    A residual network for image classification.

    This mirrors the CIFAR-10 ResNet architecture:
        Stem -> 3 stages of ResBlocks -> GlobalAvgPool -> Linear classifier

    Args:
        num_blocks:   Number of ResBlocks per stage (depth control)
        in_channels:  Number of input channels (3 for RGB, 1 for grayscale)
        num_classes:  Number of output classes
    """

    def __init__(self, num_blocks, in_channels=3, num_classes=10):
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ---- Stem: initial convolution to expand channels ----
        # Input: [N, in_channels, H, W]
        # Output: [N, 16, H, W]  (same spatial size, 16 channels)
        #
        # PyTorch: self.stem = nn.Sequential(
        #     nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.stem = cudalearn.Sequential()
        self.stem.add("conv", cudalearn.Conv2d(in_channels, 16, 3,
                                                padding=1, stride=1,
                                                use_bias=False))
        self.stem.add("bn", cudalearn.BatchNorm2d(16))
        self.stem.add("relu", cudalearn.ReLU())

        # ---- Stage 1: N blocks at 16 channels, no downsampling ----
        # Input: [N, 16, H, W]   Output: [N, 16, H, W]
        # PyTorch: self.stage1 = self._make_stage(16, 16, num_blocks, stride=1)
        self.stage1_blocks = self._make_stage(16, 16, num_blocks, stride=1)

        # ---- Stage 2: N blocks at 32 channels, downsample 2x ----
        # Input: [N, 16, H, W]   Output: [N, 32, H/2, W/2]
        # First block strides by 2 (halves spatial dims), rest stride by 1.
        # PyTorch: self.stage2 = self._make_stage(16, 32, num_blocks, stride=2)
        self.stage2_blocks = self._make_stage(16, 32, num_blocks, stride=2)

        # ---- Stage 3: N blocks at 64 channels, downsample 2x ----
        # Input: [N, 32, H/2, W/2]   Output: [N, 64, H/4, W/4]
        # PyTorch: self.stage3 = self._make_stage(32, 64, num_blocks, stride=2)
        self.stage3_blocks = self._make_stage(32, 64, num_blocks, stride=2)

        # ---- Classification head ----
        # GlobalAvgPool2d: [N, 64, H/4, W/4] -> [N, 64]
        # Linear:          [N, 64]            -> [N, num_classes]
        #
        # PyTorch: self.gap = nn.AdaptiveAvgPool2d(1)
        #          self.fc = nn.Linear(64, num_classes)
        self.head_bn = cudalearn.BatchNorm2d(64)
        self.head_relu = cudalearn.ReLU()
        self.gap = cudalearn.GlobalAvgPool2d()
        self.fc = cudalearn.Linear(64, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a list of ResBlocks for one stage.

        Args:
            in_channels:  Input channels for the first block
            out_channels: Output channels for all blocks in this stage
            num_blocks:   Number of ResBlocks to create
            stride:       Stride for the FIRST block (others use stride=1)

        Returns:
            list[ResBlock]: The blocks for this stage

        The first block may downsample (stride=2) and change channels.
        Remaining blocks maintain the same spatial size and channels.
        """
        blocks = []

        # First block: may downsample and change channels
        # PyTorch: blocks.append(ResBlock(in_channels, out_channels, stride))
        blocks.append(ResBlock(in_channels, out_channels, stride))

        # Remaining blocks: same channels, stride=1
        for _ in range(1, num_blocks):
            # PyTorch: blocks.append(ResBlock(out_channels, out_channels, 1))
            blocks.append(ResBlock(out_channels, out_channels, stride=1))

        return blocks

    def parameters(self):
        """Collect all learnable parameters from every part of the model."""
        params = []
        # Stem parameters (conv weight, BN gamma, BN beta)
        params += list(self.stem.parameters())
        # Stage 1-3 parameters
        for block in self.stage1_blocks + self.stage2_blocks + self.stage3_blocks:
            params += block.parameters()
        # Head parameters
        params += list(self.head_bn.parameters())
        # head_relu has no parameters
        # gap has no parameters
        params += list(self.fc.parameters())
        return params

    def print_summary(self):
        """
        Print a detailed model summary showing each component and its
        parameter count. Similar to torchsummary or model.print().
        """
        print("=" * 65)
        print("  ResNet Architecture Summary")
        print("=" * 65)
        print(f"  Config: {self.num_blocks} blocks/stage, "
              f"{self.in_channels} input channels, "
              f"{self.num_classes} classes")
        print("-" * 65)

        total = 0

        # Stem
        stem_params = sum(p.size for p in self.stem.parameters())
        print(f"  Stem: Conv({self.in_channels}->16, 3x3) + BN + ReLU")
        print(f"         Parameters: {stem_params}")
        total += stem_params

        # Stages
        for stage_idx, (stage_name, blocks) in enumerate([
            ("Stage 1 (16 ch)", self.stage1_blocks),
            ("Stage 2 (32 ch)", self.stage2_blocks),
            ("Stage 3 (64 ch)", self.stage3_blocks),
        ], 1):
            stage_params = 0
            print(f"\n  {stage_name}:")
            for i, block in enumerate(blocks):
                bp = block.num_params()
                stage_params += bp
                print(f"    Block {i}: {block.summary_str()}")
                print(f"              Parameters: {bp}")
            print(f"    Stage total: {stage_params}")
            total += stage_params

        # Head
        head_bn_params = sum(p.size for p in self.head_bn.parameters())
        fc_params = sum(p.size for p in self.fc.parameters())
        head_total = head_bn_params + fc_params
        print(f"\n  Head: BN(64) + ReLU + GAP + Linear(64->{self.num_classes})")
        print(f"         BN parameters:     {head_bn_params}")
        print(f"         Linear parameters: {fc_params}")
        print(f"         Head total:        {head_total}")
        total += head_total

        print("-" * 65)
        print(f"  TOTAL PARAMETERS: {total:,}")
        print("=" * 65)

        return total


# =============================================================================
# Main: Create a ResNet and run a forward pass with dummy data
# =============================================================================

def main():
    print("=" * 65)
    print("  ResNet Preview: Python frontend for cudalearn")
    print("=" * 65)
    print()

    # ---- Configuration ----
    # We create a small ResNet suitable for low-resolution images.
    # num_blocks=2 gives us a "ResNet-14" style architecture:
    #   1 (stem conv) + 2*2 (stage1) + 2*2 (stage2) + 2*2 (stage3) + 1 (fc) = 14 layers
    #
    # PyTorch: model = ResNet(num_blocks=2, in_channels=4, num_classes=10)
    num_blocks = 2
    in_channels = 4   # Using 4 channels for demonstration (not RGB)
    num_classes = 10

    print(f"Creating ResNet with {num_blocks} blocks/stage, "
          f"{in_channels} input channels, {num_classes} classes...")
    print()

    # ---- Create the model ----
    model = ResNet(num_blocks=num_blocks,
                   in_channels=in_channels,
                   num_classes=num_classes)

    # ---- Print model summary ----
    total_params = model.print_summary()
    print()

    # ---- Create dummy input ----
    # Shape: [batch=2, channels=4, height=8, width=8]
    # This is small enough to run quickly but exercises the full architecture.
    #
    # PyTorch: x = torch.randn(2, 4, 8, 8)
    batch_size = 2
    height = 8
    width = 8
    print(f"Creating dummy input: [{batch_size}, {in_channels}, {height}, {width}]")
    x = cudalearn.randn(batch_size, in_channels, height, width)
    print(f"  Input tensor: {x}")
    print()

    # ---- Forward pass through the full architecture ----
    # We manually chain the components since our ResNet is not a single Sequential.
    # In PyTorch, model(x) would handle this via the forward() method.
    print("Running forward pass...")
    print("-" * 65)

    # Stem: Conv -> BN -> ReLU
    # [2, 4, 8, 8] -> [2, 16, 8, 8]
    # PyTorch: out = self.stem(x)
    out = model.stem.forward(x)
    print(f"  After stem:    {out}")

    # Stage 1: 2 ResBlocks at 16 channels
    # [2, 16, 8, 8] -> [2, 16, 8, 8]  (no downsampling)
    #
    # NOTE: Our cudalearn library does not have a built-in residual addition
    # operation exposed to Python. In a full implementation, we would need
    # element-wise tensor addition: out = relu(main_path(x) + skip(x)).
    #
    # For this preview, we demonstrate the forward pass through the main
    # path of each block, showing the architecture and parameter counts.
    # The full residual implementation will be done in C++ in Chapter 19.
    #
    # PyTorch: out = self.stage1(out)
    for i, block in enumerate(model.stage1_blocks):
        out = block.main_path.forward(out)
        print(f"  After stage1 block {i}: {out}")

    # Stage 2: 2 ResBlocks, first one downsamples 2x
    # [2, 16, 8, 8] -> [2, 32, 4, 4]
    # PyTorch: out = self.stage2(out)
    for i, block in enumerate(model.stage2_blocks):
        out = block.main_path.forward(out)
        print(f"  After stage2 block {i}: {out}")

    # Stage 3: 2 ResBlocks, first one downsamples 2x
    # [2, 32, 4, 4] -> [2, 64, 2, 2]
    # PyTorch: out = self.stage3(out)
    for i, block in enumerate(model.stage3_blocks):
        out = block.main_path.forward(out)
        print(f"  After stage3 block {i}: {out}")

    # Head: BN -> ReLU -> GlobalAvgPool -> Linear
    # [2, 64, 2, 2] -> BN -> ReLU -> [2, 64] -> [2, 10]
    # PyTorch: out = self.head_bn(out); out = F.relu(out)
    out = model.head_bn.forward(out)
    out = model.head_relu.forward(out)
    print(f"  After head BN+ReLU: {out}")

    # PyTorch: out = self.gap(out).view(out.size(0), -1)
    out = model.gap.forward(out)
    print(f"  After GAP:     {out}")

    # PyTorch: out = self.fc(out)
    out = model.fc.forward(out)
    print(f"  Final output:  {out}")

    # ---- Inspect the output ----
    # Convert to numpy and check the logits
    logits_np = out.to_numpy()
    print(f"\n  Output logits (numpy):")
    print(f"    shape: {logits_np.shape}")
    print(f"    values:\n{logits_np}")

    print()
    print("-" * 65)
    print("  Forward pass complete!")
    print()
    print("  NOTE: This preview runs the main path only (no skip connections).")
    print("  The full ResNet with residual additions will be implemented in")
    print("  Chapter 19 (cnn_resnet) as a C++ Module with proper forward().")
    print()
    print("  What Chapter 19 will add:")
    print("    - ResBlock as a C++ Module subclass with element-wise addition")
    print("    - Full forward() that computes main_path(x) + skip(x)")
    print("    - CIFAR-10 training with data augmentation")
    print("    - Learning rate scheduling with CosineAnnealingLR")
    print("-" * 65)


if __name__ == "__main__":
    main()

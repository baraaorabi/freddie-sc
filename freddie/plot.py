from collections import defaultdict
from typing import Union

from freddie.segment import CanonIntervals, aln_t

import numpy.typing as npt
import numpy as np
from matplotlib import colormaps  # type: ignore
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt


def color_aln_matrix(
    read_classes: list[str],
    aln_matrix: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], mpl_colors.ListedColormap, mpl_colors.BoundaryNorm,]:
    classes_list = sorted(set(cs for cs in read_classes))
    aln_to_color_val = dict()
    color_to_val = defaultdict(lambda: len(color_to_val))
    for i, cs in enumerate(classes_list):
        color = (1.0, 1.0, 1.0, 1.0)  # white
        aln_to_color_val[(cs, aln_t.unaln)] = color_to_val[(1.0, 1.0, 1.0, 1.0)]

        color = (0.0, 0.0, 0.0, 1.0)  # black
        aln_to_color_val[(cs, aln_t.polyA)] = color_to_val[color]

        color = colormaps["tab10"](i % len(colormaps["tab10"].colors))
        aln_to_color_val[(cs, aln_t.exon)] = color_to_val[color]

        color = color[:3] + (0.5,)  # half transparent
        aln_to_color_val[(cs, aln_t.intron)] = color_to_val[color]
    aln_colors, aln_color_bounds = list(
        zip(
            *sorted(
                color_to_val.items(),
                key=lambda x: x[1],
            )
        )
    )
    aln_color_bounds = aln_color_bounds + (len(aln_color_bounds),)
    aln_color_bounds = [x - 0.1 for x in aln_color_bounds]
    cmap = mpl_colors.ListedColormap(aln_colors)
    norm = mpl_colors.BoundaryNorm(
        aln_color_bounds,
        cmap.N,  # type: ignore
    )

    matrix = aln_matrix.copy()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = aln_to_color_val[(read_classes[i], matrix[i, j])]
    return matrix, cmap, norm


def plot_cints(
    cints: CanonIntervals,
    unique: bool = True,
    min_height: int = 5,
    figsize: tuple[int, int] = (15, 10),
    out_prefix: Union[str, None] = None,
    read_bins: Union[tuple[list[int], ...], None] = None,
    read_classes: Union[list[str], None] = None,
    read_types: Union[list[tuple[str, ...]], None] = None,
):
    """
    Plot the intervals and the matrix representation of the intervals using matplotlib's imshow

    Parameters
    ----------
    unique : bool, optional
        if True plot only unique rows of the matrix
    min_height : int, optional
        intervals shorter than min_height are plotted in red, otherwise in blue
    out_prefix : str, optional
        if not None save the plot to out_prefix.png and out_prefix.pdf
    """
    if read_classes == None:
        read_classes = [""] * len(cints.reads)
    if read_types == None:
        read_types = [(tuple())] * len(cints.reads)
    if read_bins == None:
        read_bins = ([rid for rid in range(len(cints.reads))],)
    N = sum(len(read_bin) for read_bin in read_bins)
    fig, axes = plt.subplots(
        nrows=len(read_bins) + 1,
        ncols=2,
        figsize=figsize,
        sharex="col",
        sharey="row",
        gridspec_kw={
            "height_ratios": [1] + [5 * len(read_bin) / N for read_bin in read_bins],
            "width_ratios": [10, 1],
        },
        squeeze=False,
    )
    fig.suptitle(
        f"Classes {len(set(read_classes))}, "
        + f"Types {len(set(read_types))}, "
        + f"Reads {len(cints.reads)}"
    )

    plt.subplots_adjust(wspace=0, hspace=0)
    heights_ax = axes[0, 0]
    fig.subplots_adjust(hspace=0)
    heights = (
        [0] + [interval.end - interval.start for interval in cints.intervals] + [0]
    )
    heights_ax.bar(
        np.arange(0, len(heights), 1),
        heights,
        width=1,
        color=["red" if h < min_height else "blue" for h in heights],
    )
    heights_ax.set_ylabel("Interval length", size=10)
    heights_ax.set_ylim(0, 50)
    yticks = np.arange(5, 50 + 1, 5)
    heights_ax.set_yticks(yticks)
    heights_ax.set_yticklabels(yticks, size=8)
    heights_ax.grid()

    full_aln_matrix, aln_cmap, aln_norm = color_aln_matrix(
        read_classes, cints.get_matrix()
    )
    celltypes, full_celltype_matrix = get_celltype_matrix(cints, read_types)
    full_matrix = np.concatenate((full_aln_matrix, full_celltype_matrix), axis=1)
    for imshow_axes, read_bin in zip(axes[1:, :], read_bins):
        aln_ax = imshow_axes[0]
        ct_ax = imshow_axes[1]
        if len(read_bin) == 0:
            continue
        matrix = full_matrix[read_bin, :]
        if unique:
            matrix = np.unique(matrix, axis=0)
        aln_matrix = matrix[:, : full_aln_matrix.shape[1]]
        celltype_matrix = matrix[:, full_aln_matrix.shape[1] :]

        unique_read_count = aln_matrix.shape[0]
        aln_ax.imshow(
            aln_matrix,
            cmap=aln_cmap,
            norm=aln_norm,
            aspect="auto",
            interpolation="none",
        )
        aln_ax.set_ylabel(f"n={len(read_bin)}, u={unique_read_count}", size=10)
        aln_ax.set_xlabel("Interval index", size=10)
        starts = (
            [0]
            + [interval.start for interval in cints.intervals]
            + [cints.intervals[-1].end]
        )
        xticks = np.arange(1, len(starts), max(1, len(starts) // 30))
        if xticks[-1] != len(starts) - 1:
            xticks = np.append(xticks, len(starts) - 1)
        aln_ax.set_xticks(xticks - 0.5)
        aln_ax.set_xticklabels(
            [f"{i}) {starts[i]:,}" for i in xticks],
            size=8,
            rotation=90,
        )
        yticks = np.arange(0, unique_read_count, max(1, unique_read_count // 30))
        aln_ax.set_yticks(yticks - 0.5)
        aln_ax.set_yticklabels(yticks.astype(int), size=8)
        aln_ax.grid(which="major", axis="both")

        ct_ax.imshow(
            celltype_matrix, cmap="binary", aspect="auto", interpolation="none"
        )
        xticks = np.arange(0, len(celltypes))
        ct_ax.set_xticks(xticks + 0.5)
        ct_ax.tick_params(
            axis="both",
            which="major",
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        ct_ax.set_xticks(xticks, labels=celltypes, size=8, rotation=90, minor=True)
        ct_ax.grid(which="major", axis="both")

    corner_ax = axes[0, 1]
    corner_ax.tick_params(
        axis="both",
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )
    for _, spine in corner_ax.spines.items():
        spine.set_visible(False)
    plt.tight_layout()
    if out_prefix is not None:
        plt.savefig(f"{out_prefix}.png", dpi=500, bbox_inches="tight")
        plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.show()


def get_celltype_matrix(
    cints: CanonIntervals, read_types: list[tuple[str, ...]]
) -> tuple[list[str], npt.NDArray[np.uint8]]:
    celltypes: list[str] = sorted({ct for cts in read_types for ct in cts})
    full_celltype_matrix: npt.NDArray[np.uint8] = np.zeros(
        shape=(len(cints.reads), len(celltypes)),
        dtype=np.uint8,
    )
    for i, cts in enumerate(read_types):
        for ct in cts:
            full_celltype_matrix[i, celltypes.index(ct)] = celltypes.index(ct) + 1
    return celltypes, full_celltype_matrix

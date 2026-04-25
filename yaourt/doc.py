import os
import tempfile
import subprocess
import sys
from typing import Optional, List, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh


def get_latest_pypi_version(  # pragma: no cover
    package_name="yet-another-onnxruntime-extensions",
) -> str:
    """Returns the latest published version."""

    import requests

    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url, timeout=10)

    assert response.status_code == 200, f"Unable to retrieve the version response={response}"
    data = response.json()
    version = data["info"]["version"]
    return version


def update_version_package(
    version: str, package_name="yet-another-onnxruntime-extensions"
) -> str:
    "Adds dev if the major version is different from the latest published one."
    released = get_latest_pypi_version(package_name)
    shorten_r = ".".join(released.split(".")[:2])
    shorten_v = ".".join(version.split(".")[:2])
    return version if shorten_r == shorten_v else f"{shorten_v}.dev"


def plot_legend(
    text: str, text_bottom: str = "", color: str = "green", fontsize: int = 15
) -> "matplotlib.axes.Axes":  # noqa: F821
    """
    Plots a graph with only text (for sphinx-gallery).

    :param text: legend
    :param text_bottom: text at the bottom
    :param color: color
    :param fontsize: font size
    :return: axis
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot()
    ax.axis([0, 5, 0, 5])
    ax.text(2.5, 4, "END", fontsize=10, horizontalalignment="center")
    ax.text(
        2.5,
        2.5,
        text,
        fontsize=fontsize,
        bbox={"facecolor": color, "alpha": 0.5, "pad": 10},
        horizontalalignment="center",
        verticalalignment="center",
    )
    if text_bottom:
        ax.text(4.5, 0.5, text_bottom, fontsize=7, horizontalalignment="right")
    ax.grid(False)
    ax.set_axis_off()
    return ax


def rotate_align(ax, angle=15, align="right"):
    """Rotates x-label and aligns them to the right. Returns ax."""
    for label in ax.get_xticklabels():
        label.set_rotation(angle)
        label.set_horizontalalignment(align)
    return ax


def save_fig(ax, name: str, **kwargs) -> "matplotlib.axis.Axis":  # noqa: F821
    """Applies ``tight_layout`` and saves the figures. Returns ax."""
    fig = ax.get_figure()
    fig.savefig(name, **kwargs)
    return ax


def title(ax: "plt.axes", title: str) -> "matplotlib.axis.Axis":  # noqa: F821
    "Adds a title to axes and returns them."
    ax.set_title(title)
    return ax


def plot_histogram(
    tensor: np.ndarray,
    ax: Optional["plt.axes"] = None,  # noqa: F821
    bins: int = 30,
    color: str = "orange",
    alpha: float = 0.7,
) -> "matplotlib.axis.Axis":  # noqa: F821
    "Computes the distribution for a tensor."
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.cla()
    ax.hist(tensor, bins=bins, color=color, alpha=alpha)
    ax.set_yscale("log")
    return ax


def _find_in_PATH(prog: str) -> Optional[str]:
    """
    Looks into every path mentioned in ``%PATH%`` a specific file,
    it raises an exception if not found.

    :param prog: program to look for
    :return: path
    """
    sep = ";" if sys.platform.startswith("win") else ":"
    path = os.environ["PATH"]
    for p in path.split(sep):
        f = os.path.join(p, prog)
        if os.path.exists(f):
            return p
    return None


def _run_subprocess(args: List[str], cwd: Optional[str] = None):
    assert not isinstance(args, str), "args should be a sequence of strings, not a string."

    p = subprocess.Popen(
        args, cwd=cwd, shell=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Use communicate() to read stdout and stderr concurrently, avoiding the
    # deadlock that occurs when the subprocess fills its stderr pipe buffer
    # while the main process is blocked waiting for stdout data.
    stdout_data, stderr_data = p.communicate()
    stdout_output = stdout_data.decode(errors="ignore")
    error = stderr_data.decode(errors="ignore")
    raise_exception = any(
        phrase in stdout_output
        for phrase in ("fatal error", "CMake Error", "gmake: ***", "): error C", ": error: ")
    )
    if error and raise_exception:
        raise RuntimeError(
            f"An error was found in the output. The build is stopped."
            f"\n{stdout_output}\n---\n{error}"
        )
    return stdout_output + "\n" + error


def _run_graphviz(filename: str, image: str, engine: str = "dot") -> str:
    """
    Runs Graphviz.

    :param filename: filename which contains the graph definition
    :param image: output image
    :param engine: *dot* or *neato*
    :return: output of graphviz
    """
    ext = os.path.splitext(image)[-1]
    assert ext in {
        ".png",
        ".bmp",
        ".fig",
        ".gif",
        ".ico",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".ps",
        ".svg",
        ".vrml",
        ".tif",
        ".tiff",
        ".wbmp",
    }, f"Unexpected extension {ext!r} for {image!r}."
    assert not sys.platform.startswith("win"), "this is not working on Windows"
    exe = engine
    if os.path.exists(image):
        os.remove(image)
    cmd = [exe, f"-T{ext[1:]}", filename, "-o", image]
    output = _run_subprocess(cmd)
    assert os.path.exists(image), (
        f"Unable to find {image!r}, command line is "
        f"{' '.join(cmd)!r}, Graphviz failed due to\n{output}"
    )
    return output


def draw_graph_graphviz(dot: Union[str, onnx.ModelProto], image: str, engine: str = "dot") -> str:
    """
    Draws a graph using Graphviz.

    :param dot: dot graph or ModelProto
    :param image: output image file path
    :param engine: *dot* or *neato*
    :return: Graphviz output
    """
    if isinstance(dot, onnx.ModelProto):
        from onnx.tools.net_drawer import GetOpNodeProducer, GetPydotGraph

        pydot_graph = GetPydotGraph(
            dot.graph,
            name=dot.graph.name,
            rankdir="TB",
            node_producer=GetOpNodeProducer("docstring"),
        )
        sdot = pydot_graph.to_string()
    elif isinstance(dot, str):
        if "{" not in dot:
            assert dot.endswith(".onnx"), f"Unexpected file extension for {dot!r}"
            proto = onnx.load(dot)
            return draw_graph_graphviz(proto, image, engine=engine)
        sdot = dot
    else:
        raise TypeError(f"Unexpected type {type(dot)} for dot.")
    assert "{" in sdot, f"This string is not a dot string\n{sdot}"
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(sdot.encode("utf-8"))
        fp.close()

        filename = fp.name
        assert os.path.exists(
            filename
        ), f"File {filename!r} cannot be created to store the graph."
        out = _run_graphviz(filename, image, engine=engine)
        assert os.path.exists(
            image
        ), f"Graphviz failed with no reason, {image!r} not found, output is {out}."
        os.remove(filename)
        return out


def plot_dot(
    dot: Union[str, onnx.ModelProto],
    ax: Optional["matplotlib.axis.Axis"] = None,  # noqa: F821
    engine: str = "dot",
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.axis.Axis":  # noqa: F821
    """
    Draws a dot graph into a matplotlib axis.

    :param dot: dot graph or ModelProto
    :param ax: optional matplotlib axis; if None, a new figure and axis are created
    :param engine: *dot* or *neato*
    :param figsize: size of the figure if *ax* is None
    :return: matplotlib axis containing the rendered graph image
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 1, figsize=figsize)
        clean = True
    else:
        clean = False

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        fp.close()

        draw_graph_graphviz(dot, fp.name, engine=engine)
        img = np.asarray(Image.open(fp.name))
        os.remove(fp.name)

        ax.imshow(img)

    if clean:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_axis_off()
        ax.get_figure().tight_layout()
    return ax


def plot_text(
    text: str,
    ax: Optional["plt.axes"] = None,  # noqa: F821
    title: str = "",
    fontsize: int = 6,
    line_color_map: Optional[dict] = None,
    default_color: str = "#333333",
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.axis.Axis":  # noqa: F821
    """
    Renders a block of text as a matplotlib figure, with optional per-line
    color coding based on the first character of each line.

    :param text: the text to render (newlines split into rows)
    :param ax: optional matplotlib axis; if *None* a new figure and axis are created
    :param title: optional axis title
    :param fontsize: font size for the rendered text
    :param line_color_map: mapping from a line's first character to a color
        string (e.g. ``{"+": "green", "-": "red", "@": "blue"}``).
        Lines whose first character is not in the map use *default_color*.
    :param default_color: color for lines not matched by *line_color_map*
    :param figsize: ``(width, height)`` in inches; only used when *ax* is *None*
    :return: the matplotlib axis
    """
    import matplotlib.pyplot as plt

    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    if ax is None:
        if figsize is None:
            figsize = (10, max(2, n_lines * 0.18 + 0.5))
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_lines)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 2, loc="left", pad=3)

    color_map = line_color_map or {}
    for i, line in enumerate(lines):
        color = color_map.get(line[:1], default_color)
        ax.text(
            0.01,
            n_lines - i - 0.5,
            line,
            fontsize=fontsize,
            color=color,
            fontfamily="monospace",
            va="center",
            transform=ax.transData,
        )

    return ax


def hhistograms_data() -> dict:
    """
    Returns sample benchmark data suitable for use with :func:`hhistograms`.

    The dictionary can be passed directly to ``pandas.DataFrame`` to produce
    a DataFrame with the columns expected by :func:`hhistograms`.

    :return: a dictionary of lists usable as ``pandas.DataFrame`` input
    """
    return {
        "average": {
            0: 0.0020172329999877544,
            1: 0.002073898999997254,
            2: 0.0022592019999956394,
            3: 0.002121683000009398,
            4: 0.002126841000008426,
            5: 0.002168457999991915,
            6: 0.002244355000002542,
            7: 0.0022011400000064895,
            8: 0.002154946999999084,
            9: 0.0021154199999955385,
            10: 0.0020685299999922795,
            11: 0.0022017719999985275,
            12: 0.0020702290000008364,
            13: 0.0021156809999911277,
            14: 0.0021835879999980534,
            15: 0.0020887549999952172,
            16: 0.000551501000004464,
            17: 0.00068478600000617,
        },
        "min_exec": {
            0: 0.0017108299999563314,
            1: 0.001987460000009378,
            2: 0.001773719999982859,
            3: 0.00204343999998855,
            4: 0.001976730000023963,
            5: 0.0017028299999765294,
            6: 0.0020138200000019424,
            7: 0.002151430000003529,
            8: 0.0020578200000272774,
            9: 0.0018366799999967043,
            10: 0.001885519999996177,
            11: 0.0019714099999873723,
            12: 0.0016516000000137865,
            13: 0.001985019999983706,
            14: 0.001757259999976668,
            15: 0.0019274100000075123,
            16: 0.0005356799999844952,
            17: 0.0005467899999985093,
        },
        "max_exec": {
            0: 0.0021741499999734515,
            1: 0.002246560000003228,
            2: 0.00343564999998307,
            3: 0.0022677800000110437,
            4: 0.002236210000000938,
            5: 0.0032833400000072286,
            6: 0.002555330000041067,
            7: 0.002272720000019035,
            8: 0.002228419999983089,
            9: 0.002193380000016987,
            10: 0.0021609799999623646,
            11: 0.0026991000000180067,
            12: 0.0022127400000044872,
            13: 0.0024491199999829407,
            14: 0.0024486799999976936,
            15: 0.0022466699999768025,
            16: 0.0005794500000320113,
            17: 0.0008353600000191363,
        },
        "name": {
            0: "80,128,50,1,1,0",
            1: "80,128,25,1,1,0",
            2: "80,64,50,1,1,0",
            3: "80,64,25,1,1,0",
            4: "40,128,50,1,1,0",
            5: "40,128,25,1,1,0",
            6: "40,64,50,1,1,0",
            7: "40,64,25,1,1,0",
            8: "80,128,50,1,1,0",
            9: "80,128,25,1,1,0",
            10: "80,64,50,1,1,0",
            11: "80,64,25,1,1,0",
            12: "40,128,50,1,1,0",
            13: "40,128,25,1,1,0",
            14: "40,64,50,1,1,0",
            15: "40,64,25,1,1,0",
            16: "40,64,25,1,1,0",
            17: "baseline",
        },
        "input": {
            0: "sparse",
            1: "sparse",
            2: "sparse",
            3: "sparse",
            4: "sparse",
            5: "sparse",
            6: "sparse",
            7: "sparse",
            8: "sparse",
            9: "sparse",
            10: "sparse",
            11: "sparse",
            12: "sparse",
            13: "sparse",
            14: "sparse",
            15: "sparse",
            16: "dense",
            17: "dense",
        },
    }


def hhistograms(
    df: "pandas.DataFrame",  # noqa: F821
    keys: Union[str, Tuple[str, ...]] = "name",
    metric: str = "average",
    baseline: str = "baseline",
    title: str = "Benchmark",
    limit: int = 50,
    ax: Optional["matplotlib.axis.Axis"] = None,  # noqa: F821
) -> "matplotlib.axis.Axis":  # noqa: F821
    """
    Plots horizontal histograms with error bars for benchmark comparisons.

    Shows the *limit* best-performing configurations alongside the baseline.
    Inspired by ``onnx_extended.plotting.benchmark.hhistograms``.

    :param df: DataFrame containing benchmark results; must have columns
        matching *keys* and *metric*, plus ``min_exec`` and ``max_exec``
        columns when they exist (used for the x-axis limits)
    :param keys: column name(s) to group by; the last key identifies the
        baseline row via *baseline*
    :param metric: column name holding the primary performance metric
    :param baseline: substring used to identify the baseline row in the last
        *keys* column
    :param title: chart title
    :param limit: maximum number of non-baseline rows to display
    :param ax: existing matplotlib axis; a new figure is created when *None*
    :return: the matplotlib axis
    """
    import pandas

    if not isinstance(keys, (tuple, list)):
        keys = (keys,)

    dfm = (
        df[[*keys, metric]].groupby(list(keys), as_index=False).agg(["mean", "min", "max"]).copy()
    )
    if dfm.shape[1] == 3:
        dfm = dfm.reset_index(drop=False)
    dfm.columns = [*keys, metric, "min", "max"]
    dfi = dfm.sort_values(metric).reset_index(drop=True)
    base = dfi[dfi[keys[-1]].str.contains(baseline)]
    not_base = dfi[~dfi[keys[-1]].str.contains(baseline)].reset_index(drop=True)
    if not_base.shape[0] > limit:
        not_base = not_base[:limit]
    merged = pandas.concat([base, not_base], axis=0)
    merged = merged.sort_values(metric).reset_index(drop=True).set_index(list(keys))

    fig = None
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, max(1, merged.shape[0] / 2)))

    err_min = merged[metric] - merged["min"]
    err_max = merged["max"] - merged[metric]
    merged[[metric]].plot.barh(ax=ax, title=title, xerr=[err_min, err_max])
    b = df.loc[df[keys[-1]] == baseline, metric].mean()
    ax.plot([b, b], [0, df.shape[0]], "r--")
    x_min = df[metric].min()
    x_max = df[metric].max()
    if "min_exec" in df.columns:
        x_min = (df["min_exec"].min() + x_min) / 2
    ax.set_xlim([x_min, x_max])

    if fig is not None:
        fig.tight_layout()
    return ax


def demo_mlp_model(filename: str) -> onnx.ModelProto:
    """
    Returns a demo MLP model as an ONNX ModelProto.

    :param filename: unused, kept for API compatibility
    :return: an ONNX ModelProto representing a small MLP network
    """
    return oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["x", "p_layers_0_weight::T10"], ["_onx_matmul_x"]),
                oh.make_node("Add", ["_onx_matmul_x", "layers.0.bias"], ["linear"]),
                oh.make_node("Relu", ["linear"], ["relu"]),
                oh.make_node("MatMul", ["relu", "p_layers_2_weight::T10"], ["_onx_matmul_relu"]),
                oh.make_node("Add", ["_onx_matmul_relu", "layers.2.bias"], ["output_0"]),
            ],
            "experiment",
            [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (3, 10))],
            [oh.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (3, 1))],
            [
                onh.from_array(
                    np.random.randn(10, 32).astype(np.float32), name="p_layers_0_weight::T10"
                ),
                onh.from_array(
                    np.random.randn(32, 1).astype(np.float32), name="p_layers_2_weight::T10"
                ),
                onh.from_array(np.random.randn(32).astype(np.float32), name="layers.0.bias"),
                onh.from_array(
                    np.array([-0.1422213315963745], dtype=np.float32), name="layers.2.bias"
                ),
            ],
        ),
        functions=[],
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=8,
    )

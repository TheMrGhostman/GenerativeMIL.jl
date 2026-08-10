using CairoMakie

"""
    scatter_by_class(X, y; kwargs...)

Scatter plot of the (2 × n) matrix `X` colored by the labels in `y`.
Each class is plotted separately so that the legend shows one entry
(color + class number) per class.
"""
function scatter_by_class(X, y;
    title = "", xlabel = "comp 1", ylabel = "comp 2",
    legend_title = "class", colormap = :tab10,
    markersize = 6, legend_markersize = 16, resolution = (800, 600)
)
    @assert size(X, 1) == 2 "X has to be a (2 × n) matrix"
    @assert size(X, 2) == length(y) "number of columns of X and length of y have to match"

    classes = sort(unique(y))
    colors = cgrad(colormap, max(length(classes), 2), categorical = true)

    f = Figure(size = resolution)
    ax = Axis(f[1, 1], xlabel = xlabel, ylabel = ylabel, title = title)

    for (i, c) in enumerate(classes)
        idx = findall(==(c), y)
        scatter!(
            ax, X[1, idx], X[2, idx],
            color = colors[i], markersize = markersize, label = string(c)
        )
    end

    legend_elements = [
        MarkerElement(color = colors[i], marker = :circle, markersize = legend_markersize)
        for i in eachindex(classes)
    ]
    Legend(f[1, 2], legend_elements, string.(classes), legend_title, framevisible = false)
    return f, ax
end

"""
    scatter3d_by_class(X, y; kwargs...)

Scatter plot of the (3 × n) matrix `X` colored by the labels in `y`.
Each class is plotted separately so that the legend shows one entry
(color + class number) per class.
"""
function scatter3d_by_class(X, y;
    title = "", xlabel = "comp 1", ylabel = "comp 2", zlabel = "comp 3",
    legend_title = "class", colormap = :tab10,
    markersize = 6, legend_markersize = 16, resolution = (800, 600)
)
    @assert size(X, 1) == 3 "X has to be a (3 × n) matrix"
    @assert size(X, 2) == length(y) "number of columns of X and length of y have to match"

    classes = sort(unique(y))
    colors = cgrad(colormap, max(length(classes), 2), categorical = true)

    f = Figure(size = resolution)
    ax = Axis3(f[1, 1], xlabel = xlabel, ylabel = ylabel, zlabel = zlabel, title = title)

    for (i, c) in enumerate(classes)
        idx = findall(==(c), y)
        scatter!(
            ax, X[1, idx], X[2, idx], X[3, idx],
            color = colors[i], markersize = markersize, label = string(c)
        )
    end

    legend_elements = [
        MarkerElement(color = colors[i], marker = :circle, markersize = legend_markersize)
        for i in eachindex(classes)
    ]
    Legend(f[1, 2], legend_elements, string.(classes), legend_title, framevisible = false)
    return f, ax
end

"""
    heatmap_by_class(D, y; kwargs...)

Heatmap of a (n × n) pairwise distance/similarity matrix `D` with a colorbar
and the class labels `y` (length n) used as the x and y tick labels, so that
each cell can be identified by (row class, column class).
"""
function heatmap_by_class(
    D, y;
    title = "", colormap = :viridis, colorbar_label = "distance",
    resolution = (700, 600), show_values = true, value_fmt = x -> string(round(x, digits = 2)),
    value_fontsize = 12,
)
    n = size(D, 1)
    @assert size(D, 1) == size(D, 2) "D has to be a square matrix"
    @assert length(y) == n "length of y has to match the size of D"

    f = Figure(size = resolution)
    ax = Axis(
        f[1, 1], title = title,
        xticks = (1:n, string.(y)), yticks = (1:n, string.(y)),
        xlabel = "class", ylabel = "class", yreversed = true,
    )

    hm = heatmap!(ax, 1:n, 1:n, permutedims(D), colormap = colormap)
    Colorbar(f[1, 2], hm, label = colorbar_label)

    if show_values
        lo, hi = extrema(D)
        mid = (lo + hi) / 2
        for i in 1:n, j in 1:n
            v = D[i, j]
            text!(
                ax, j, i, text = value_fmt(v),
                align = (:center, :center), fontsize = value_fontsize,
                color = v > mid ? :black : :white,
            )
        end
    end

    return f, ax
end

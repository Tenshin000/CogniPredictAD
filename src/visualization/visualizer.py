from typing import Optional, List, Sequence, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """
    Visualizer is a plotting class for exploratory data analysis (EDA).

    Usage:
        viz = Visualizer(dataset)
        viz.histogram(x="AGE", title="Age distribution", xlabel="age", ylabel="Patient Count")
        viz.scatter_plot(x="height", y="weight", hue="gender", title="Height vs Weight")
    """
    # ----------------------------#
    #        INITIALIZATION       #
    # ----------------------------#
    def __init__(self, dataset: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        :param dataset: pandas DataFrame to visualize
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame")
        sns.set_theme(style="whitegrid")
        self.dataset = dataset.copy()

    # ----------------------------#
    #           HELPERS           #
    # ----------------------------#
    def _validate_column(self, col: Optional[str], dtype: Optional[str] = None, optional: bool = False) -> Optional[str]:
        """
        Validate a single column name. If col is None, attempt to infer a column
        matching dtype ('numeric'|'categorical') or ask the user for input.
        """
        if col is None:
            if optional:
                return None
            if dtype == 'numeric':
                numerics = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
                if len(numerics) == 1:
                    return numerics[0]
                if len(numerics) > 1:
                    print(f"Multiple numeric columns found, using '{numerics[0]}' by default. Provide col explicitly to override.")
                    return numerics[0]
            elif dtype == 'categorical':
                cats = self.dataset.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                if len(cats) == 1:
                    return cats[0]
                if len(cats) > 1:
                    print(f"Multiple categorical columns found, using '{cats[0]}' by default. Provide col explicitly to override.")
                    return cats[0]
            # last resort: interactive (useful in notebooks)
            col_in = input(f"Please enter column name (available: {list(self.dataset.columns)}): ").strip()
            if col_in == "" and optional:
                return None
            if col_in not in self.dataset.columns:
                raise ValueError(f"Column '{col_in}' not found in DataFrame")
            return col_in
        else:
            if col not in self.dataset.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            return col

    def _validate_columns(self, cols: Optional[Sequence[str]], dtype: Optional[str] = None, min_count: int = 1) -> List[str]:
        """
        Ensure cols (list) are present; if None, try to infer columns with dtype.
        """
        if cols is None:
            if dtype == 'numeric':
                numerics = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
                if len(numerics) < min_count:
                    raise ValueError(f"Need at least {min_count} numeric columns; found {len(numerics)}")
                return numerics[:max(min_count, len(numerics))]
            elif dtype == 'categorical':
                cats = self.dataset.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                if len(cats) < min_count:
                    raise ValueError(f"Need at least {min_count} categorical columns; found {len(cats)}")
                return cats[:min_count]
            else:
                cols_found = list(self.dataset.columns[:min_count])
                if len(cols_found) < min_count:
                    raise ValueError("Not enough columns to infer")
                return cols_found
        else:
            cols_list = list(cols)
            for c in cols_list:
                if c not in self.dataset.columns:
                    raise ValueError(f"Column '{c}' not in DataFrame")
            if len(cols_list) < min_count:
                raise ValueError(f"Expected at least {min_count} columns, got {len(cols_list)}")
            return cols_list

    # ----------------------------#
    #            PLOTS            #
    # ----------------------------# 
    def line_plot(self, x: Optional[str] = None, y: Optional[Union[str, List[str]]] = None,
                  hue: Optional[str] = None, title: Optional[str] = None,
                  xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                  figsize=(8, 5), return_ax: bool = False, **kwargs):
        """
        Create a line plot for one or multiple numeric series.
        y can be a single column or list of columns (multiple series).
        If y is None and there are multiple numeric columns, plot them all.

        Parameters
        ----------
        x : str, optional
            Column name for the x-axis. If None, uses the DataFrame index.
        y : str or list of str, optional
            One or more numeric columns to plot. If None, all numeric columns are used.
        hue : str, optional
            Column name for grouping (different colors per category).
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        figsize : tuple, default (8, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Additional arguments passed to seaborn/matplotlib.
        """
        # infer x if appropriate
        if x is None:
            if pd.api.types.is_datetime64_any_dtype(self.dataset.index):
                x = None  # plotting will use df.index
            else:
                x = self._validate_column(None, dtype=None, optional=True)

        if isinstance(y, list):
            ycols = [self._validate_column(c, dtype='numeric') for c in y]
        elif isinstance(y, str):
            ycols = [self._validate_column(y, dtype='numeric')]
        else:
            ycols = self._validate_columns(None, dtype='numeric', min_count=1)

        fig, ax = plt.subplots(figsize=figsize)
        if hue:
            # seaborn handles grouping by hue
            sns.lineplot(data=self.dataset, x=x, y=ycols[0] if len(ycols) == 1 else None, hue=hue, ax=ax, **kwargs)
            if len(ycols) > 1:
                for c in ycols:
                    sns.lineplot(data=self.dataset, x=x, y=c, label=c, ax=ax, **kwargs)
        else:
            for c in ycols:
                if x is None:
                    ax.plot(self.dataset[c], label=c, **kwargs)
                else:
                    ax.plot(self.dataset[x], self.dataset[c], label=c, **kwargs)
            ax.legend()

        ax.set_title(title or "Line Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x or "index")
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(", ".join(ycols))
        fig.tight_layout()
        if return_ax:
            return ax
        

    def scatter_plot(self, x: Optional[str] = None, y: Optional[str] = None, hue: Optional[str] = None,
                     size: Optional[str] = None, style: Optional[str] = None,
                     title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                     figsize=(7, 6), return_ax: bool = False, **kwargs):
        """
        Create a scatter plot with optional grouping and styling.

        Parameters
        ----------
        x, y : str
            Numeric columns for x- and y-axis.
        hue : str, optional
            Column used to color points by category.
        size : str, optional
            Column used to scale point sizes.
        style : str, optional
            Column used to set marker style.
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        figsize : tuple, default (7, 6)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Additional seaborn styling arguments.
        """
        x = self._validate_column(x, dtype='numeric')
        y = self._validate_column(y, dtype='numeric')
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(data=self.dataset, x=x, y=y, hue=hue, size=size, style=style, ax=ax, **kwargs)
        ax.set_title(title or "Scatter Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
        fig.tight_layout()
        if return_ax:
            return ax
        

    def histogram(self, x: Optional[str] = None, bins: int = 30, kde: bool = False, rug: bool = False,
                  title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                  figsize=(7, 5), stacked: bool = False, by: Optional[str] = None, return_ax: bool = False, **kwargs):
        """
        Create a histogram of a numeric column.

        Parameters
        ----------
        x : str
            Numeric column to plot.
        bins : int, default 30
            Number of histogram bins.
        kde : bool, default False
            Whether to add a kernel density estimate.
        rug : bool, default False
            Whether to add a rug plot.
        stacked : bool, default False
            If True, stack histograms by category when using 'by'.
        by : str, optional
            Categorical column to group histograms.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (7, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        x = self._validate_column(x, dtype='numeric')
        fig, ax = plt.subplots(figsize=figsize)
        if by:
            sns.histplot(data=self.dataset, x=x, hue=by, bins=bins, kde=kde, multiple="stack" if stacked else "layer", ax=ax, **kwargs)
        else:
            sns.histplot(data=self.dataset, x=x, bins=bins, kde=kde, ax=ax, **kwargs)
        if rug:
            sns.rugplot(data=self.dataset, x=x, ax=ax)
        ax.set_title(title or "Histogram")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("count")
        fig.tight_layout()
        if return_ax:
            return ax
        

    def bar_plot(self, x: Optional[str] = None, y: Optional[str] = None, estimator=np.mean, ci='sd',
                 title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                 figsize=(8, 5), annotate: bool = False, return_ax: bool = False, **kwargs):
        """
        Bar plot for categorical x vs numerical y. If y is None, draws frequency bars (countplot).

        Parameters
        ----------
        x : str
            Categorical column.
        y : str, optional
            Numeric column. If None, plot counts instead of aggregated values.
        estimator : function, default np.mean
            Function to aggregate numeric values (e.g., np.mean, np.median).
        ci : str, default 'sd'
            Error bar type ('sd', 'ci', or None).
        annotate : bool, default False
            If True, annotate bars with values.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (8, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        if y is None:
            x = self._validate_column(x, dtype='categorical')
            # handle palette/hue deprecation: if palette but no hue -> color by x
            hue = kwargs.pop('hue', None)
            legend_flag = kwargs.pop('legend', None)
            if "palette" in kwargs and hue is None:
                hue = x
                if legend_flag is None:
                    legend_flag = False
            fig, ax = plt.subplots(figsize=figsize)
            sns.countplot(data=self.dataset, x=x, hue=hue, ax=ax, **kwargs)
            # remove legend if requested
            if legend_flag is False and ax.get_legend() is not None:
                ax.get_legend().remove()
            if annotate:
                for p in ax.patches:
                    height = p.get_height()
                    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
        else:
            x = self._validate_column(x, dtype='categorical')
            y = self._validate_column(y, dtype='numeric')
            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(data=self.dataset, x=x, y=y, estimator=estimator, errorbar="sd", ax=ax, **kwargs)
            if annotate:
                for p in ax.patches:
                    height = p.get_height()
                    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        ax.set_title(title or "Bar Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("count" if y is None else y)
        fig.tight_layout()
        if return_ax:
            return ax
        

    def box_plot(self, x: Optional[str] = None, y: Optional[str] = None, by: Optional[str] = None,
                 title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                 figsize=(8, 5), return_ax: bool = False, **kwargs):
        """
        Box plot. For categorical x and numeric y. If x is None and y provided, x is inferred.
        If both None, draws boxplots for all numeric columns.
        
        Parameters
        ----------
        x : str, optional
            Categorical column. If None, draw boxplots for all numeric columns.
        y : str, optional
            Numeric column to plot.
        by : str, optional
            Additional grouping variable.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (8, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        if x is None and y is None:
            numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns to plot")
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=self.dataset[numeric_cols], ax=ax, **kwargs)
            ax.set_title(title or "Box Plot")
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            fig.tight_layout()
            if return_ax:
                return ax
            
        else:
            if y is None:
                # Assume the user passed x as numeric and wants per-column boxes
                y = x
                x = None
            x_valid = self._validate_column(x, dtype='categorical', optional=True)
            y_valid = self._validate_column(y, dtype='numeric')
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=self.dataset, x=x_valid, y=y_valid, hue=y_valid, ax=ax, **kwargs)
            ax.set_title(title or "Box Plot")
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(x_valid or "")
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(y_valid)
            fig.tight_layout()
            if return_ax:
                return ax
            

    def violin_plot(self, x: Optional[str] = None, y: Optional[str] = None, hue: Optional[str] = None,
                    title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                    figsize=(8, 5), return_ax: bool = False, **kwargs):
        """
        Create a violin plot to show numeric distribution by category.

        Parameters
        ----------
        x : str, optional
            Categorical column.
        y : str
            Numeric column.
        hue : str, optional
            Column for nested grouping.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (8, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        x_valid = self._validate_column(x, dtype='categorical', optional=True)
        y_valid = self._validate_column(y, dtype='numeric')
        fig, ax = plt.subplots(figsize=figsize)
        sns.violinplot(data=self.dataset, x=x_valid, y=y_valid, hue=hue, ax=ax, **kwargs)
        ax.set_title(title or "Violin Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x_valid or "")
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y_valid)
        fig.tight_layout()
        if return_ax:
            return ax

    def kde_plot(self, x: Optional[str] = None, shade: bool = True, bw_method: Optional[float] = None,
                 title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                 figsize=(7, 5), return_ax: bool = False, **kwargs):
        """
        Create a kernel density estimate (KDE) plot for a numeric column.

        Parameters
        ----------
        x : str
            Numeric column to plot.
        shade : bool, default True
            Whether to fill the area under the curve.
        bw_method : float, optional
            Bandwidth for the KDE.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (7, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        x = self._validate_column(x, dtype='numeric')
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(data=self.dataset, x=x, fill=shade, bw_method=bw_method, ax=ax, **kwargs)
        ax.set_title(title or "KDE Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("density")
        fig.tight_layout()
        if return_ax:
            return ax
        

    def pair_plot(self, cols: Optional[Sequence[str]] = None, hue: Optional[str] = None,
                  diag_kind: str = 'hist', corner: bool = False, title: Optional[str] = None, **kwargs):
        """
        Create pairwise plots for multiple numeric variables.

        Parameters
        ----------
        cols : list of str, optional
            Numeric columns to include. If None, use all numeric columns.
        hue : str, optional
            Categorical column for color encoding.
        diag_kind : {'hist', 'kde'}, default 'hist'
            Type of plot for the diagonal.
        corner : bool, default False
            If True, only plot lower triangle.
        title : str, optional
            Figure title.
        kwargs : dict
            Extra seaborn arguments.

        Returns
        -------
        seaborn.axisgrid.PairGrid
            The seaborn PairGrid object.
        """
        cols_list = list(self._validate_columns(cols, dtype='numeric', min_count=2))
        plot_cols = cols_list + ([hue] if hue and hue not in cols_list else [])
        g = sns.pairplot(self.dataset[plot_cols], hue=hue, diag_kind=diag_kind, corner=corner, **kwargs)
        if title:
            plt.suptitle(title, y=1.02)
        else:
            plt.suptitle("Pair Plot", y=1.02)
        plt.tight_layout()
        return g


    def heatmap_corr(self, cols: Optional[Sequence[str]] = None, method: str = 'pearson',
                     annot: bool = True, cmap: Optional[str] = None, title: Optional[str] = None,
                     xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                     figsize=(8, 6), fmt: str = ".2f", return_ax: bool = False):
        """
        Create a correlation heatmap for numeric variables.

        Parameters
        ----------
        cols : list of str, optional
            Numeric columns to include. If None, use all numeric columns.
        method : {'pearson', 'spearman', 'kendall'}, default 'pearson'
            Correlation method.
        annot : bool, default True
            Whether to display correlation values.
        cmap : str, optional
            Colormap for heatmap.
        fmt : str, default '.2f'
            Format for annotation values.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (8, 6)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        """

        cols_list = self._validate_columns(cols, dtype='numeric', min_count=2) if cols is not None else self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        corr = self.dataset[cols_list].corr(method=method)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, vmin=-1, vmax=1, ax=ax)
        ax.set_title(title or f"Correlation Heatmap ({method})")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        fig.tight_layout()
        if return_ax:
            return ax
        

    def joint_plot(self, x: Optional[str] = None, y: Optional[str] = None, kind: str = 'scatter',
                   title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                   figsize=(7, 7), **kwargs):
        """
        Create a joint plot showing the relationship between two variables.

        Parameters
        ----------
        x, y : str
            Numeric columns.
        kind : {'scatter', 'kde', 'hex', 'reg'}, default 'scatter'
            Type of joint plot.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (7, 7)
            Size of the figure (height maps to jointplot 'height').
        kwargs : dict
            Extra seaborn arguments.

        Returns
        -------
        seaborn.axisgrid.JointGrid
            The seaborn JointGrid object.
        """
        x = self._validate_column(x, dtype='numeric')
        y = self._validate_column(y, dtype='numeric')
        # seaborn's jointplot uses "height" for size: map figsize[0] to height
        g = sns.jointplot(data=self.dataset, x=x, y=y, kind=kind, height=figsize[0], **kwargs)
        g.fig.suptitle(title or "Joint Plot")
        if xlabel:
            g.ax_joint.set_xlabel(xlabel)
        if ylabel:
            g.ax_joint.set_ylabel(ylabel)
        plt.tight_layout()
        return g

    def count_plot(self, x: Optional[str] = None, hue: Optional[str] = None, figsize=(8, 5),
                   title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                   annotate: bool = False, return_ax: bool = False, **kwargs):
        """
        Create a count plot for categorical frequencies.

        Parameters
        ----------
        x : str
            Categorical column.
        hue : str, optional
            Column for grouping counts.
        annotate : bool, default False
            If True, add counts above each bar.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (8, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        x = self._validate_column(x, dtype='categorical')
        # Extract and manage 'legend' from kwargs (so it is not passed to seaborn)
        legend_flag = kwargs.pop('legend', None)
        # If a palette is passed but no hue, force hue=x and default to no legend
        if "palette" in kwargs and hue is None:
            hue = x
            if legend_flag is None:
                legend_flag = False

        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(data=self.dataset, x=x, hue=hue, ax=ax, **kwargs)
        if legend_flag is False and ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_title(title or "Count Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("count")

        if annotate:
            # annotate bars with heights
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

        fig.tight_layout()
        if return_ax:
            return ax


    def strip_swarm_plot(self, x: Optional[str] = None, y: Optional[str] = None, kind: str = 'strip',
                         title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                         figsize=(8, 5), return_ax: bool = False, **kwargs):
        """
        Create a strip plot or swarm plot for categorical vs numeric data.

        Parameters
        ----------
        x : str, optional
            Categorical column.
        y : str
            Numeric column.
        kind : {'strip', 'swarm'}, default 'strip'
            Type of plot.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (8, 5)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra seaborn arguments.
        """
        if kind not in ('strip', 'swarm'):
            raise ValueError("kind must be 'strip' or 'swarm'")
        x_valid = self._validate_column(x, dtype='categorical', optional=True)
        y_valid = self._validate_column(y, dtype='numeric')
        fig, ax = plt.subplots(figsize=figsize)
        if kind == 'strip':
            sns.stripplot(data=self.dataset, x=x_valid, y=y_valid, ax=ax, **kwargs)
        else:
            sns.swarmplot(data=self.dataset, x=x_valid, y=y_valid, ax=ax, **kwargs)
        ax.set_title(title or f"{kind.capitalize()} Plot")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        fig.tight_layout()
        if return_ax:
            return ax
        

    def pie_chart(self, x: Optional[str] = None, labels: Optional[Sequence[str]] = None,
                  explode: Optional[Sequence[float]] = None, figsize=(6, 6), autopct: Optional[str] = '%1.1f%%',
                  title: Optional[str] = None, return_ax: bool = False):
        """
        Create a pie chart from categorical frequencies.

        Parameters
        ----------
        x : str
            Categorical column.
        labels : list of str, optional
            Labels for the pie slices.
        explode : list of float, optional
            Offsets for each slice.
        autopct : str, default '%1.1f%%'
            Format string for percentages.
        figsize : tuple, default (6, 6)
            Figure size.
        title : str, optional
            Chart title.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        """
        x = self._validate_column(x, dtype='categorical')
        counts = self.dataset[x].value_counts()
        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(counts.values, labels=labels or counts.index.astype(str), autopct=autopct, explode=explode)
        ax.set_title(title or f"Pie chart of {x}")
        ax.axis('equal')
        fig.tight_layout()
        if return_ax:
            return ax
        

    def area_plot(self, x: Optional[str] = None, y: Optional[Sequence[str]] = None, stacked: bool = False,
                  title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                  figsize=(10, 6), return_ax: bool = False, **kwargs):
        """
        Create an area plot for one or more numeric series.

        Parameters
        ----------
        x : str, optional
            Column for x-axis (default uses index).
        y : list of str, optional
            Numeric columns to plot. If None, use all numeric columns.
        stacked : bool, default False
            If True, stack the areas.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (10, 6)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra pandas/matplotlib arguments.
        """
        if y is None:
            ycols = self.dataset.select_dtypes(include=[np.number]).columns.tolist()
            if not ycols:
                raise ValueError("No numeric columns for area plot")
        else:
            ycols = list(self._validate_columns(y, dtype='numeric', min_count=1))
        fig, ax = plt.subplots(figsize=figsize)
        self.dataset[ycols].plot.area(stacked=stacked, ax=ax, **kwargs)
        ax.set_title(title or "Area Plot")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        fig.tight_layout()
        if return_ax:
            return ax
        

    def hexbin_plot(self, x: Optional[str] = None, y: Optional[str] = None, gridsize: int = 30,
                    title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                    figsize=(7, 6), return_ax: bool = False, **kwargs):
        """
        Create a hexbin plot for two numeric variables.

        Parameters
        ----------
        x, y : str
            Numeric columns.
        gridsize : int, default 30
            Number of hexagons across the x-axis.
        title, xlabel, ylabel : str, optional
            Plot title and axis labels.
        figsize : tuple, default (7, 6)
            Figure size.
        return_ax : bool, default False
            If True, return the matplotlib Axes.
        kwargs : dict
            Extra matplotlib arguments.
        """
        x = self._validate_column(x, dtype='numeric')
        y = self._validate_column(y, dtype='numeric')
        fig, ax = plt.subplots(figsize=figsize)
        hb = ax.hexbin(self.dataset[x], self.dataset[y], gridsize=gridsize, cmap='Blues', **kwargs)
        fig.colorbar(hb, ax=ax, label='counts')
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title or "Hexbin Plot")
        fig.tight_layout()
        if return_ax:
            return ax
        

    def correlation_scatter_matrix(self, cols: Optional[Sequence[str]] = None, figsize=(8, 8)):
        """
        Create a scatter matrix (pairwise scatter plots with KDE on diagonals).

        Parameters
        ----------
        cols : list of str, optional
            Numeric columns to include. If None, use all numeric columns.
        figsize : tuple, default (8, 8)
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib Figure object.
        """
        cols_list = self._validate_columns(cols, dtype='numeric', min_count=2) if cols is not None else self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        fig = pd.plotting.scatter_matrix(self.dataset[cols_list], diagonal='kde', figsize=figsize)
        plt.suptitle("Scatter Matrix")
        plt.tight_layout()
        return fig

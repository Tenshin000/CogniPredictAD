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
        viz.histogram(x="age", title="Age distribution", xlabel="age", ylabel="count")
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

        :param col: column name or None
        :param dtype: 'numeric', 'categorical', or None
        :param optional: if True, returns None when col not provided
        :return: validated column name or None
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
        Generic line plot. y can be a single column or list of columns (multiple series).
        If y is None and there are multiple numeric columns, plot them all.
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
        Scatter plot with optional hue, size, style.
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
        Histogram. If 'by' is provided (categorical), draws multiple histograms by category.
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
        annotate: if True and y is None, annotate bars with counts; if y provided, annotates bar heights.
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
            sns.barplot(data=self.dataset, x=x, y=y, estimator=estimator, ci=ci, ax=ax, **kwargs)
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
                # assume the user passed x as numeric and wants per-column boxes
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
        Violin plot for distribution of numeric y by categorical x.
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
        Kernel Density Estimate plot for a single numeric column.
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
        Pairplot (pairwise relationships). If cols is None, uses numeric columns.
        Returns the PairGrid object.
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
        Correlation heatmap for numeric columns (or subset).
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
        Joint plot for two numeric variables. kind can be 'scatter', 'kde', 'hex', 'reg'.
        Returns the JointGrid / FacetGrid object.
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
        Countplot for categorical variable frequencies.
        If a palette is provided but hue is None, the method sets hue=x and hides the legend by default
        (avoids seaborn FutureWarning when palette is used without hue).
        annotate: if True, adds counts above each bar.
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
        Strip or Swarm plot. kind='strip' or 'swarm'.
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
        Pie chart of value counts for a categorical column.
        Returns the figure (pie charts are not typical Axes-returners).
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
        Area plot for time series or numeric series. y can be multiple columns.
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
        Hexbin plot for two numeric variables (useful for large scatter density).
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
        Scatter matrix (pairwise scatter) colored by density via KDE shading in the diagonal.
        Returns the Figure.
        """
        cols_list = self._validate_columns(cols, dtype='numeric', min_count=2) if cols is not None else self.dataset.select_dtypes(include=[np.number]).columns.tolist()
        fig = pd.plotting.scatter_matrix(self.dataset[cols_list], diagonal='kde', figsize=figsize)
        plt.suptitle("Scatter Matrix")
        plt.tight_layout()
        

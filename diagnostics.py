"""Regression diagnostics."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import statsmodels.api as sm


def residual_plot(expected, predicted, residuals, ax=None):
    """
    Plot fitted values against residuals.

    Together with the scatter plot of fitted values vs residuals, add a
    Locally Weighted Scatterplot Smoothing (LOWESS) regression line.
    The top 3 y-axis values are also annotated.

    You can look for heteroscedasticity (whether the assumption of
    homoscedasticity is not violated).  The scatter plot of fitted
    values against the residuals should show no clear pattern.  A
    common problem of heteroscedasticity is that the variances increases
    with the mean, and we may see a fan-shaped pattern.

    Parameters
    ----------
    expected : vector
        Expected values, the response variable.

    predicted : vector
        Predicted values, the predictor variable.

    residuals : vector
        Residual values, used to make annotations of the largest
        absolute residuals.

    ax : matplotlib axis, optional
        Plot into this axis, otherside otherwise grab the current
        axis or make a new one if not existing.
    """
    # Residual scatter plot with smooth lowess.
    ax = seaborn.residplot(
        x=predicted,
        y=expected,
        lowess=True,
        scatter_kws=dict(alpha=0.7, s=10),
        line_kws=dict(color='red', lw=1, alpha=0.8),
        ax=ax
    )
    ax.set_title('Residuals × Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    # Annotations
    abs_resid = np.abs(residuals).sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        ax.text(predicted[i], residuals[i], i, size=8)


def qq_plot(std_residuals, ax=None):
    """
    Plot quantiles of the normalized residuals.

    The Q–Q plot has use for regression with assumption that the errors
    are normally distributed.  This is the basic assumption for linear
    regression: if the normalized residuals do not follow a normal
    distribution, the interpretation may be affected and the model may
    have a weaken inference.

    The Q–Q-plot depicts the standardized residuals (z-scores) against
    theoretical quantiles of the normal distribution.  Ideally, the
    points should all lie near the 1:1 line (the diagonal line,
    intercept 0 and slope 1).  If the pattern is S-shaped,
    banana-shaped, or too off the diagonal line, you may need to fit a
    different model to the data.

    The top 3 y-axis values are also annotated.

    Parameters
    ----------
    std_residuals : vector
        Vector of the standardized residuals (z-scores).

    ax : matplotlib, optional
        Plot into this axis, otherside otherwise grab the current
        axis or make a new one if not existing.

    See Also
    --------
    std_residual_hist : histogram of normalized residuals.

    References
    ----------
    Crawley (2007) The R Book (1st ed.). Wiley Publishing.

    James, Witten, Hastie & Tibshirani (2014) An Introduction to
    Statistical Learning: With Applications in R. Springer Publishing
    Company, Incorporated.
    """
    qq = sm.ProbPlot(std_residuals)
    fig = qq.qqplot(ax=ax, line='45', alpha=0.7, color='#4C72B0',
                    lw=1, markersize=3)
    if not ax:
        ax = fig.axes[0]
    ax.set_title('Normal Q–Q')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Standardized residuals')
    # Annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(std_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for rank, i in enumerate(abs_norm_resid_top_3):
        x = np.flip(qq.theoretical_quantiles, 0)[rank]
        y = std_residuals[i]
        ax.text(x, y, i, size=8)


def std_residual_hist(std_residuals, **distplot_kws):
    """
    Plot histogram of normalized residuals.

    The top 3 x-axis values are also annotated with vertical lines.

    Parameters
    ----------
    std_residuals : vector
        The vector of normalized residuals.

    distplot_kws : dict, optional
        Dictionary of parameters for `seaborn.distplot`, including
        the matplotlib axis.

    References
    ----------
    James, Witten, Hastie & Tibshirani (2014) An Introduction to
    Statistical Learning: With Applications in R. Springer Publishing
    Company, Incorporated.
    """
    ax = seaborn.distplot(std_residuals, **distplot_kws)
    ax.set_title('Histogram of normalized residuals')
    ax.set_xlabel('Normalized residual value')
    ax.set_ylabel('Density')
    # Annotations
    abs_sq_norm_resid = np.sqrt(np.abs(std_residuals))
    desc_order = np.flip(np.argsort(abs_sq_norm_resid), 0)
    top_3 = desc_order[:3]
    _, ypos = ax.get_ylim()
    for i in top_3:
        ax.axvline(std_residuals[i], color='gray', linestyle='--',
                   linewidth=0.5)
        ypos *= 0.9
        ax.text(std_residuals[i], ypos, i, color='gray', size=8)


def scale_location_plot(predicted, std_residuals, ax=None):
    """
    Plot the fitted values with the absoluted squared rooted residuals.

    This is a positive-valued version of the Residual × Fitted plot
    (the `residual_plot` method).  The residuals should be randomly
    distributed around the horizontal line.  It’s somewhat more
    sensitive to outliers.

    Along the scatter plot of fitted values against the absolute of
    square-rooted standardized residuals, includes a regression
    line of the Locally Weighted Scatterplot Smoothing (LOWESS). The
    top 3 y-axis values are also annotated.

    Parameters
    ----------
    predicted : vector
        Vector of predicted values, the predictor variable.

    std_residuals : vector
        Vector of normalized residuals (z-score).

    ax : matplotlib axis, optional
        Plot into this axis, otherside otherwise grab the current
        axis or make a new one if not existing.

    References
    ----------
    James, Witten, Hastie & Tibshirani (2014) An Introduction to
    Statistical Learning: With Applications in R. Springer Publishing
    Company, Incorporated.
    """
    if not ax:
        ax = plt.figure().gca()
    abs_sq_sd_resid = np.sqrt(np.abs(std_residuals))
    ax.scatter(predicted, abs_sq_sd_resid, alpha=0.8, s=10)
    seaborn.regplot(
        x=predicted,
        y=abs_sq_sd_resid,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws=dict(color='red', lw=1, alpha=0.8),
        ax=ax
    )
    ax.set_title('Scale–Location')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel(r'$\sqrt{|Standardized\ residuals|}$')
    # Annotations
    desc_order = np.flip(np.argsort(abs_sq_sd_resid), 0)
    top_3 = desc_order[:3]
    for i in top_3:
        ax.text(predicted[i], abs_sq_sd_resid[i], i, size=8)


def cook_plot(cooks, ax=None):
    """
    Plot a vertical line of Cook’s distance for each data point.

    The stem plot of Cook’s distance will help you finding outliers.

    Parameters
    ----------
    cooks : vector
        Vector of Cook’s distances.

    ax : matplotlib axis, optional
        Plot into this axis, otherside otherwise grab the current
        axis or make a new one if not existing.

    See Also
    --------
    `cookleverage_plot` for a definition of Cook’s distance.
    """
    if not ax:
        ax = plt.figure().gca()
    unused_ml, stems, baseline = ax.stem(cooks, markerfmt=' ')
    baseline.set_linewidth(0.1)
    for stem in stems:
        stem.set_linewidth(1)
    ax.set_xlim((0, cooks.shape[0]))
    ax.set_title('Cook’s distance')
    ax.set_xlabel('Observation number')
    ax.set_ylabel('Cook’s distance')
    ymax = np.max(cooks)
    if ymax > 0.5:
        ax.axhline(0.5, color='red', linewidth=0.5, linestyle='--')
    if ymax > 1.0:
        ax.axhline(1.0, color='red', linewidth=0.5, linestyle='--')
    # Annotations
    leverage_top_3 = np.flip(np.argsort(cooks), 0)[:3]
    for i in leverage_top_3:
        ax.text(i, cooks[i], i, size=8)


def leverage_plot(leverage, n_params, ax=None):
    """
    Plot stems of leverage for each observation.

    This will plot the leverage value for each data point, together
    with a horizontal line that marks the rule-of-thumb for highly
    influential data points.

    Parameters
    ----------
    leverage : vector
        Vector of leverage values.

    n_params : int
        Number of parameters in the model.  This is used to compute the
        rule-of-thumb for a highly influential data point.

    ax : matplotlib axis, optional
        Plot into this axis, otherside otherwise grab the current
        axis or make a new one if not existing.

    References
    ----------
    Crawley (2007) The R Book (1st ed.). Wiley Publishing.
    """
    if not ax:
        ax = plt.figure().gca()
    # Rule of thumb: above this line, points are influential.
    from math import ceil, log
    threshold = 2 * n_params / leverage.shape[0]
    ax.axhline(threshold, color='gray', linewidth=1, linestyle='--')
    thres = str(round(threshold, ceil(log(1 / threshold))))
    ax.text(0, threshold * 1.05, thres, color='gray', size=8)
    unused_ml, stems, baseline = ax.stem(leverage, markerfmt=' ')
    baseline.set_linewidth(0.1)
    for stem in stems:
        stem.set_linewidth(1)
    ax.set_xlim((0, leverage.shape[0]))
    ax.set_title('Leverage')
    ax.set_xlabel('Observation number')
    ax.set_ylabel('Leverage')
    # Annotations
    top_3 = np.flip(np.argsort(leverage), 0)[:3]
    for i in top_3:
        ax.text(i, leverage[i], i, size=8)


def residuals_leverage_plot(leverage, std_residuals, cooks, n_params, ax=None):
    r"""
    Plot the leverages of data points with Cook’s distance contours.

    Plot a scatter plot of the leverage values and the standardized
    residuals (z-score).  A regression line of the Locally
    Weighted Scatterplot Smoothing (LOWESS).  Also, this plots contour
    lines for equal Cook’s distance.

    Points increase in influence to the extent that they lie on their
    own, far away from the mean value of $x$.  Measures of leverage
    for a data point are proportional to $(x - \bar{x})^2$.  A
    definition for leverage is
    \begin{equation}
        h_i =   \frac{1}{n}
              + \frac{\left(x_i - \bar{x}\right)^2}
                     {\sum \left(x_j - \bar{x}\right)}.
    \end{equation}
    where $p$ is the number of parameters in the model and $n$ is
    the number of data points.  A good rule of thumb is that a point is
    highly influential if it is leverage
    \begin{equation}
        h_i > \frac{2p}{n}.
    \end{equation}

    Parameters
    ----------
    leverage : vector
        Vector fo leverage values.

    std_residuals : vector
        Vector of standardized residuals (z-score).

    cooks : vector
        Vector of Cook’s distances.

    n_params : int
        The number of parameters of the model.  Used to plot contour
        lines of the Cook’s distance.

    See Also
    --------
    `cookleverage_plot` for a definition of leverage.
    """
    if not ax:
        ax = plt.figure().gca()
    ax.axhline(0, c='gray', linestyle='--', alpha=0.5, lw=0.7)
    ax.scatter(leverage, std_residuals, alpha=0.7, s=10)
    seaborn.regplot(
        x=leverage,
        y=std_residuals,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws=dict(color='red', lw=1, alpha=0.8),
        ax=ax
    )
    ax.set_xlim((0, np.max(leverage) * 1.1))
    ax.set_ylim((np.min(std_residuals)-0.5, np.max(std_residuals)+0.5))
    ax.set_title('Residuals × Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    # Annotations
    leverage_top_3 = np.flip(np.argsort(cooks), 0)[:3]
    for i in leverage_top_3:
        ax.text(leverage[i], std_residuals[i], i, size=8)

    def contour(formula, x, label=None):
        """Shenanigans for Cook’s distance contours."""
        ax.plot(x, formula(x), label=label, lw=1, ls='--', color='red')

    # C = 0.5 line
    contour(
        formula=lambda x: np.sqrt((0.5 * n_params * (1 - x)) / x),
        x=np.linspace(0.001, 0.200, 50),
        label='Cook’s distance'
    )
    # C = 1 line
    contour(
        formula=lambda x: np.sqrt((1 * n_params * (1 - x)) / x),
        x=np.linspace(0.001, 0.200, 50)
    )
    ax.legend(loc='upper right')


def cook_leverage_plot(leverage, cooks, ax=None):
    r"""
    Plot the Cook’s distance.

    The Cook’s distance helps identifying influential data points by
    measuring the effect of deleting an observation, which is given by
    computing the difference with and without a particular point.

    Data points may be influential because of the large residauls
    (outliers) and/or high leverage.  Leverage refers to the fact that
    points increase in influence the further they are away from the
    mean value of `x` and from the other values of `x` (the extent they
    lie on their own).

    Data points may be influential because of the large residauls
    (outliers) and/or high leverage.  Leverage refers to the fact that
    points increase in influence the further they are away from the
    mean value of $x$ and from the other values of $x$ (the extent they
    lie on their own).

    Leverage values are usued to weight the absolute
    residuals in Cook’s distance. The leverage is given by
    \begin{equation}
        h_i =   \frac{1}{n}
              + \frac{\left(x_i - \bar{x}\right)^2}
                     {\sum \left(x_j - \bar{x}\right)}.
    \end{equation}
    A good rule of thumb is that a point is highly influential if
    it is leverage
    \begin{equation}
        h_i > \frac{2p}{n},
    \end{equation}
    where $p$ is the number of parameters in the model and $n$ is
    the number of data points.

    In page 359 of Crawley (2007) the Cook’s distance is given by
    \begin{equation}
        C_i = \lvert r_{i}^{*} \rvert
              \left(
                  \frac{n-p}{p} \frac{h_i}{1-h_i}
              \right) ^ {1/2},
    \end{equation}
    where $\lvert r_{i}^{*} \rvert$ is the absolute values of the
    deletion residuals.

    Parameters
    ----------
    cooks : vector
        Vector of Cook’s distance values.

    ax : matplotlib axis, optional
        Plot into this axis, otherside otherwise grab the current
        axis or make a new one if not existing.

    References
    ----------
    Crawley (2007) The R Book (1st ed.). Wiley Publishing.
    """
    if not ax:
        ax = plt.figure().gca()
    xmax = np.max(leverage) * 1.1
    ymax = np.max(cooks) * 1.1
    leverage = leverage / (1 - leverage)
    ax.scatter(leverage, cooks, alpha=0.7, s=10)
    seaborn.regplot(
        x=leverage,
        y=cooks,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws=dict(color='red', lw=1, alpha=0.8),
        ax=ax
    )
    for slope in np.arange(0, 10, 1):
        ax.plot([0, xmax], [0, xmax * slope],
                c='gray', linestyle='--', alpha=0.5, lw=0.5)
        x = xmax
        y = xmax * slope
        if y > ymax:
            x = ymax / slope
            y = x * slope * 0.9
            x *= 1.01  # Do after computing `y`.
        ax.text(x*0.95, y*1.01, str(slope), size=8, color='gray', alpha=0.5)
    ax.set_xlim((0 - 0.001, xmax))
    ax.set_ylim((0 - 0.001, ymax))
    ax.set_title(r'Cook’s dist × Leverage $h_i$ $h_i/(1-h_i)$')
    ax.set_xlabel('Leverage / (1 - Leverage)')
    ax.set_ylabel('Cook’s distance')
    # Annotations
    leverage_top_3 = np.flip(np.argsort(cooks), 0)[:3]
    for i in leverage_top_3:
        ax.text(leverage[i], cooks[i], i, size=8)


def lm_plot(model, expected):
    """
    Linear model regression diagnostic plots.

    This function emulates the R’s diagnostic plots, creating the
    following grid of six plots:

    |-----|-----|-----|
    | 1.1 | 1.2 | 1.3 |
    |-----|-----|-----|
    | 2.1 | 2.2 | 2.3 |
    |-----|-----|-----|

    1.1 Residuals: fitted values × residuals.
    1.2 Normal Q–Q: standardized residuals × theoretical quantiles of
                    a normal distribution.
    1.3 Scale–Location: sqaure root of the absolute residuals × fitted
                        values.
    ­2.1 Cook’s distance: observation index × Cook’s distance.
    2.2 Residuals × Leverage
    2.3 Cook’s distance × Leverage/(1 - Leverage)

    Parameters
    ----------
    model : statsmodels fitted model
        The model to obtain fitted values, residuals, standardized
        residuals, Cook’s distances, and the numbers of parameters.

    expected : vector
        Vector of expected values, the response variable.
    """
    predicted = model.fittedvalues
    residuals = model.resid
    std_residuals = model.get_influence().resid_studentized_internal
    leverage = model.get_influence().hat_matrix_diag
    n_params = len(model.params)
    cooks = model.get_influence().cooks_distance[0]
    _, ax = plt.subplots(2, 3, figsize=(12, 7), dpi=200)
    residual_plot(expected, predicted, residuals, ax=ax[0][0])
    qq_plot(std_residuals, ax=ax[0][1])
    scale_location_plot(predicted, std_residuals, ax=ax[0][2])
    cook_plot(cooks, ax=ax[1][0])
    residuals_leverage_plot(leverage, std_residuals, cooks, n_params, ax=ax[1][1])
    cook_leverage_plot(leverage, cooks, ax=ax[1][2])
    plt.tight_layout()

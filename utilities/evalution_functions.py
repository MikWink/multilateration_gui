import numpy as np

def bias(std, rmse):
    """!
    @brief  Calculate bias from std and rmse
    @param  std standard deviation.
    @param  rmse root mean square error.
    @return bias.
    """
    return rmse - std


def mean(predicitons, axis=0):
    """!
    @brief  Calculate mean
    @param  predictions estimated value.
    @param  axis dimension allong we want to estimate the standard deviation
    @return std.
    """
    if isinstance(predicitons, np.ndarray):
        return np.nanmean(predicitons, axis=axis)
    else:
        return np.nanmean(predicitons)


def std(predicitons, axis=0):
    """!
    @brief  Calculate standard deviation
    @param  predictions estimated value.
    @param  axis dimension allong we want to estimate the standard deviation
    @return std.
    """
    if isinstance(predicitons, np.ndarray):
        return np.nanstd(predicitons, axis=axis)
    else:
        return np.nanstd(predicitons)

def rmse(predictions, targets):
    """!
    @brief  Calculate root mean square error
    @param  predictions estimated value.
    @param  targets true value.
    @return rmse.
    """
    print(f'predictions: {predictions}\ntargets: {targets}')
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse
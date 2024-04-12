# WhittakerHendersonSmootherMAX_ORDER


import numpy as np

class WhittakerHendersonSmoother:
    MAX_ORDER = 5
    DIFF_COEFF = [
        [-1, 1],               # penalty on 1st derivative
        [1, -2, 1],
        [-1, 3, -3, 1],
        [1, -4, 6, -4, 1],
        [-1, 5, -10, 10, -5, 1]  # 5th derivative. 5 = MAX_ORDER
    ]
    LAMBDA_FOR_NOISE_GAIN = [0.06284, 0.0050100, 0.0004660, 4.520e-05, 4.467e-06] #trying bigger values

    def __init__(self, length, order, lambda_val):           # provide length of Numpy array, and order(penalty) and lambda(smoothing)
        self.matrix = self.make_dprime_d(order, length)
        self.times_lambda_plus_ident(self.matrix, lambda_val)
        self.cholesky_l(self.matrix)

    def smooth(self, data, out=None):                                                           # provide data to be smoothed
        if len(data) != len(self.matrix[0]):                                                    # check if data length matches matrix length
            raise ValueError(f"Data length mismatch, {len(data)} vs. {len(self.matrix[0])}")    # RAISE ERROR
        out = self.solve(self.matrix, data, out)                                                # solve the matrix equation
        return out

    @staticmethod
    def smooth_static(data, order, lambda_val):                                                 # provide data to be smoothed
        smoother = WhittakerHendersonSmoother(len(data), order, lambda_val)                     # create a new instance of the class
        return smoother.smooth(data)                                                            #adding in smooth_chunck instead of smooth

    @staticmethod
    def smooth_like_savitzky_golay(data, degree, m):                                           # provide data to be smoothed
        order = degree // 2 + 1
        bandwidth = WhittakerHendersonSmoother.savitzky_golay_bandwidth(degree, m)            # bandwidth is calculated from degree and m
        lambda_val = WhittakerHendersonSmoother.bandwidth_to_lambda(order, bandwidth)         # lambda is calculated from bandwidth
        return WhittakerHendersonSmoother.smooth_static(data, order, lambda_val)              # return the smoothed data

    @staticmethod
    def bandwidth_to_lambda(order, bandwidth):                                                # provide order and bandwidth
        if bandwidth <= 0 or bandwidth >= 0.5:                                                # check if bandwidth is within the range
            raise ValueError(f"Invalid bandwidth value: {bandwidth}")                         # check if bandwidth is within the range
        omega = 2 * np.pi * bandwidth                                                         # omega is calculated from bandwidth
        cos_term = 2 * (1 - np.cos(omega))                                                    # cos term is calculated from omega
        cos_power = cos_term                                                                  # cos power is calculated from cos term
        for _ in range(1, order):                                                             # loop through the order
            cos_power *= cos_term                                                             # multiply cos power by cos term
        lambda_val = (np.sqrt(2) - 1) / cos_power                                             # lambda is calculated from cos power
        return lambda_val                                                                     # return lambda

    @staticmethod
    def savitzky_golay_bandwidth(degree, m): 
        return 1.0 / (6.352 * (m + 0.5) / (degree + 1.379) - (0.513 + 0.316 * degree) / (m + 0.5)) # bandwidth is calculated from degree and m

    @staticmethod
    def noise_gain_to_lambda(order, noise_gain): 
        g_power = noise_gain 
        for _ in range(1, order):                                                             # loop through the order
            g_power *= noise_gain                                                             # multiply g power by noise gain
        lambda_val = WhittakerHendersonSmoother.LAMBDA_FOR_NOISE_GAIN[order] / (g_power + g_power) # lambda is calculated from g power
        return lambda_val
#The np.arange function is used to create an array of indices, and then these indices are used to select elements from the coeffs array. The np.sum function is used to calculate the sum of the products. This should be faster than the original version, especially for large arrays.
# new method below
    @staticmethod
    def make_dprime_d(order, size):
        if order < 1 or order > WhittakerHendersonSmoother.MAX_ORDER:
            raise ValueError(f"Invalid order {order}") 
        if size < order:
            raise ValueError(f"Order ({order}) must be less than number of points ({size})")

        coeffs = WhittakerHendersonSmoother.DIFF_COEFF[order - 1]
        out = [np.zeros(size - d) for d in range(order + 1)]
        for d in range(order + 1):
            for i in range((len(out[d]) + 1) // 2):
                j_range = np.arange(max(0, i - len(out[d]) + len(coeffs) - d), min(i + 1, len(coeffs) - d))
                s = np.sum(np.array(coeffs)[j_range] * np.array(coeffs)[j_range + d])
                out[d][i] = s
                out[d][len(out[d]) - 1 - i] = s
        return out



#old working method below        
#    @staticmethod
#    def make_dprime_d(order, size):
#        if order < 1 or order > WhittakerHendersonSmoother.MAX_ORDER:                          # check if order is within the range
#            raise ValueError(f"Invalid order {order}") 
#        if size < order:
#            raise ValueError(f"Order ({order}) must be less than number of points ({size})")

#        coeffs = WhittakerHendersonSmoother.DIFF_COEFF[order - 1]                             # get the coefficients for the order
#        out = [np.zeros(size - d) for d in range(order + 1)]                                  # create an empty list
#        for d in range(order + 1):                                                            # loop through the order
#            for i in range((len(out[d]) + 1) // 2):                                           # loop through the length of the list
#                s = sum(coeffs[j] * coeffs[j + d] for j in range(max(0, i - len(out[d]) + len(coeffs) - d),  
#                                                                  min(i + 1, len(coeffs) - d)))# sum the coefficients
#                out[d][i] = s                                                                  # assign the sum to the list
#                out[d][len(out[d]) - 1 - i] = s                                                # assign the sum to the list
#        return out


    @staticmethod
    def times_lambda_plus_ident(b, lambda_val):
        for i in range(len(b[0])):                                                            # loop through the length of the list
            b[0][i] = 1.0 + b[0][i] * lambda_val                                              # diagonal elements with identity added
        for d in range(1, len(b)):                                                            # loop through the length of the list
            for i in range(len(b[d])):                                                        # loop through the length of the list
                b[d][i] = b[d][i] * lambda_val                                                # off-diagonal elements


    @staticmethod
    def cholesky_l(b):
        n = len(b[0])                                                                        # get the length of the list
        dmax = len(b) - 1                                                                    # get the length of the list
        for i in range(n):                                                                   # loop through the length of the list
            for j in range(max(0, i - dmax), i + 1): 
                s = sum(b[i - k][k] * b[j - k][k] for k in range(max(0, i - dmax), j))       # sum the elements
                if i == j:                                                                   # check if i is equal to j
                    sqrt_arg = b[0][i] - s                                                   # calculate the square root argument
                    if sqrt_arg <= 0:                                                        # check if the square root argument is less than or equal to 0
                        raise RuntimeError("Cholesky decomposition: Matrix is not positive definite")
                    b[0][i] = np.sqrt(sqrt_arg)                                              # calculate the square root
                else:
                    b[i - j][j] = 1.0 / b[0][j] * (b[i - j][j] - s)                          # calculate the off-diagonal elements



    @staticmethod
    def solve(b, vec, out=None):
        if out is None or len(out) != len(vec):
            out = np.zeros(len(vec))                                                        # create an empty list
        n = len(b[0])
        dmax = len(b) - 1                                                                   # get the length of the list, set as dmax
        for i in range(n):
            s = sum(b[i - k][k] * out[k] for k in range(max(0, i - dmax), i))               # sum the elements
            out[i] = (vec[i] - s) / b[0][i]                                                 # denominator is diagonal element a[i,i]
        for i in range(n - 1, -1, -1):                                                      # loop through the length of the list
            s = sum(b[j - i][i] * out[j] for j in range(i + 1, min(i + dmax + 1, n)))       # sum the elements
            out[i] = (out[i] - s) / b[0][i]                                                 # denominator is diagonal element a[i,i]
        return out
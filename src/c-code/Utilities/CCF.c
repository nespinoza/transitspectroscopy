#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#define ARRAYD(p) ((double *) (((PyArrayObject *)p)->data)) 

/* 
 *                                [INITIALIZATION]
 * ------------------ PROTOTYPES FOR FUNCTIONS AND EXTERNAL VARIABLES -----------------------
 *
 */

//double** MakeArray(int rows, int columns);                           /* Function that makes/allocates an array of pointers    */
double* MakeVector(int nelements);                                   /* Function that makes/allocates a pointer in memory     */


/*                  [INITIALIZATION OF A METHOD]
*------------------------THE Gaussian METHOD-----------------------------
* PyObject initialization: We define a PyObject which defines a Method 
* for the CCF module: The Gaussian returns the CCF at pre-specified lags 
* back to Python. 
*------------------------------------------------------------------------
*/

static PyObject *CCF_Gaussian(PyObject *self, PyObject *args){

    // Definition of general and to-be-imported variables:
    int i,j;

    double *x, *y, *lags;
    double mu, sigma;
    int len_data, len_lags;

    PyObject *input_xarray, *input_yarray, *input_lags;

/* 
 *--------------------------------THE DATA---------------------------------------
 * After initialization of the PyObject pointers, we wish to recover the following inputs:
 *
 * input_xarray    : Vector containing the x-axis of the input data.
 * 
 * input_yarray    : Vector containing the input data.
 *
 * input_lags      : Vector containing the lags at which the CCF will be computed
 *
 * len_data        : Length of the input data (int).
 *
 * len_lags        : Length of input lag vector (int).
 *
 * mu              : Mean of the gaussian (double).
 *
 * sigma           : Standard deviation of the gaussian (double).
 * ------------------------------------------------------------------------------
*/

    // Read in python data:
    PyArg_ParseTuple(args, "OOOiidd", &input_xarray, &input_yarray, &input_lags, &len_data, &len_lags, &mu, &sigma);

    // Convert python objects back to C arrays:
    x = ARRAYD(input_xarray);
    y = ARRAYD(input_yarray);
    lags = ARRAYD(input_lags);

    // Big for loop that computes the CCF (GCCF stands for "Gaussian CCF"):
    double* GCCF = MakeVector(len_lags);
    double argument, gaussian, g_norm, g_exp_constant;

    g_norm = ( 1. / sqrt( 2. * 3.142857) ) * (1. / sigma);
    g_exp_constant = 1. / ( 2. * pow(sigma, 2) );

    for (i=0; i < len_lags; i++){

        // Compute CCF for a given lag:
        argument = 0;
        for (j=0; j < len_data; j++){

            gaussian = g_norm * exp( - ( pow(x[j] - mu - lags[i], 2) * g_exp_constant ));
            argument = argument + gaussian * y[j]; 

        }

        // Store CCF result at the given lag:
        GCCF[i] = argument;

    }
     
    // Finally, we create a Python "Object" List that contains the CCF and return it back to Python:

    PyObject *lst = PyList_New(len_lags);

    if (!lst){

       return NULL;

    }

    for (i=0; i < len_lags; i++){

        PyObject *num = PyFloat_FromDouble(GCCF[i]);
        if (!num){

            Py_DECREF(lst);
            return NULL;
        }

        PyList_SET_ITEM(lst, i, num);

    }

    free(GCCF);

    // Return object back to python:
    PyObject *MyResult = Py_BuildValue("O",lst);
    Py_DECREF(lst);
    return MyResult;

}


/*                  [INITIALIZATION OF A METHOD]
*------------------------THE DoubleGaussian METHOD-----------------------------
* PyObject initialization: We define a PyObject which defines a Method 
* for the CCF module: The DoubleGaussian returns the CCF at pre-specified lags 
* back to Python of the data against a sum of gaussians. 
*------------------------------------------------------------------------
*/

static PyObject *CCF_DoubleGaussian(PyObject *self, PyObject *args){

    // Definition of general and to-be-imported variables:
    int i,j;

    double *x, *y, *lags;
    double mu1, sigma1, mu2, sigma2;
    int len_data, len_lags;

    PyObject *input_xarray, *input_yarray, *input_lags;

/* 
 *--------------------------------THE DATA---------------------------------------
 * After initialization of the PyObject pointers, we wish to recover the following inputs:
 *
 * input_xarray     : Vector containing the x-axis of the input data.
 * 
 * input_yarray     : Vector containing the input data.
 *
 * input_lags       : Vector containing the lags at which the CCF will be computed
 *
 * len_data         : Length of the input data (int).
 *
 * len_lags         : Length of input lag vector (int).
 *
 * mu1              : Mean of the first gaussian (double).
 *
 * sigma1           : Standard deviation of the first gaussian (double).
 *
 * mu2              : Mean of the second gaussian (double).
 *
 * sigma2           : Standard deviation of the second gaussian (double).
 * ------------------------------------------------------------------------------
*/

    // Read in python data:
    PyArg_ParseTuple(args, "OOOiidddd", &input_xarray, &input_yarray, &input_lags, &len_data, &len_lags, &mu1, &sigma1, &mu2, &sigma2);

    // Convert python objects back to C arrays:
    x = ARRAYD(input_xarray);
    y = ARRAYD(input_yarray);
    lags = ARRAYD(input_lags);

    // Big for loop that computes the CCF (GCCF stands for "Double Gaussian CCF"):
    double* DGCCF = MakeVector(len_lags);
    double argument, gaussian1, gaussian2, g_norm1, g_exp_constant1, g_norm2, g_exp_constant2;

    g_norm1 = ( 1. / sqrt( 2. * 3.142857) ) * (1. / sigma1);
    g_exp_constant1 = 1. / ( 2. * pow(sigma1, 2) );

    g_norm2 = ( 1. / sqrt( 2. * 3.142857) ) * (1. / sigma2);
    g_exp_constant2 = 1. / ( 2. * pow(sigma2, 2) );

    for (i=0; i < len_lags; i++){

        // Compute CCF for a given lag:
        argument = 0;
        for (j=0; j < len_data; j++){

            gaussian1 = g_norm1 * exp( - ( pow(x[j] - mu1 - lags[i], 2) * g_exp_constant1 ));
            gaussian2 = g_norm2 * exp( - ( pow(x[j] - mu2 - lags[i], 2) * g_exp_constant2 ));

            argument = argument + (gaussian1 + gaussian2) * y[j]; 

        }

        // Store CCF result at the given lag:
        DGCCF[i] = argument;

    }
     
    // Finally, we create a Python "Object" List that contains the CCF and return it back to Python:

    PyObject *lst = PyList_New(len_lags);

    if (!lst){

       return NULL;

    }

    for (i=0; i < len_lags; i++){

        PyObject *num = PyFloat_FromDouble(DGCCF[i]);
        if (!num){

            Py_DECREF(lst);
            return NULL;
        }

        PyList_SET_ITEM(lst, i, num);

    }

    free(DGCCF);

    // Return object back to python:
    PyObject *MyResult = Py_BuildValue("O",lst);
    Py_DECREF(lst);
    return MyResult;

}

/*                  [INITIALIZATION OF A METHOD]
*------------------------THE AnyFunction METHOD-----------------------------
* PyObject initialization: We define a PyObject which defines a Method 
* for the CCF module: The AnyFunction returns the CCF of an input function evaluated 
* at pre-specified lags back to Python.
*------------------------------------------------------------------------
*/

static PyObject *CCF_AnyFunction(PyObject *self, PyObject *args){

    // Definition of general and to-be-imported variables:
    int i,j;

    double *y, *ef;
    int len_data, len_lags;

    PyObject *input_yarray, *input_evaluated_function;

/* 
 *--------------------------------THE DATA---------------------------------------
 * After initialization of the PyObject pointers, we wish to recover the following inputs:
 *
 * input_yarray             : Vector containing the input data.
 *
 * input_evaluated_function : This holds the function evaluated at different lags. This is supposed to be the flattened 
 *                            version of a matrix of dimension (len_lags, len_data), where each row contains the function 
 *                            evaluated at (x - lag). This is why we don't need input_xarray nor the input_lags here 
 *                            (its implicit in this input).
 *
 * len_data                 : Length of the input data (int).
 *
 * len_lags                 : Length of input lag vector (int).
 *
 * ------------------------------------------------------------------------------
*/

    // Read in python data:
    PyArg_ParseTuple(args, "OOii", &input_xarray, &input_evaluated_function, &len_data, &len_lags);

    // Convert python objects back to C arrays:
    y = ARRAYD(input_yarray);
    ef = ARRAYD(input_evaluated_function);

    // Big for loop that computes the CCF (AFCCF stands for "Any Function CCF"):
    double* AFCCF = MakeVector(len_lags);
    double argument;

    for (i=0; i < len_lags; i++){

        // Compute CCF for a given lag:
        argument = 0;
        for (j=0; j < len_data; j++){

            argument = argument + (ef[i * len_data + j ]) * y[j]; 

        }

        // Store CCF result at the given lag:
        AFCCF[i] = argument;

    }
     
    // Finally, we create a Python "Object" List that contains the CCF and return it back to Python:

    PyObject *lst = PyList_New(len_lags);

    if (!lst){

       return NULL;

    }

    for (i=0; i < len_lags; i++){

        PyObject *num = PyFloat_FromDouble(AFCCF[i]);
        if (!num){

            Py_DECREF(lst);
            return NULL;
        }

        PyList_SET_ITEM(lst, i, num);

    }

    free(AFCCF);

    // Return object back to python:
    PyObject *MyResult = Py_BuildValue("O",lst);
    Py_DECREF(lst);
    return MyResult;

}

static PyMethodDef CCFMethods[] = {
	{"Gaussian", CCF_Gaussian, METH_VARARGS, "Function that performs simple CCF of an input array with a gaussian."},
	{"DoubleGaussian", CCF_DoubleGaussian, METH_VARARGS, "Function that performs simple CCF of an input array with a sum of gaussians."},
	{"AnyFunction", CCF_AnyFunction, METH_VARARGS, "Function that performs simple CCF of an input array with an input set of arrays already evaluated at the lags."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef CCF =
{
    PyModuleDef_HEAD_INIT,
    "CCF",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    CCFMethods
};

/*********************************************************************
 *          [UTILITY CCF FUNCTIONS]                                  *
 *********************************************************************
 */

PyMODINIT_FUNC PyInit_CCF(void){

    return PyModule_Create(&CCF);
}

double* MakeVector(int nelements){

    // Initialize variables:
    double* Vector;
    int j;

    // Create vector:
    Vector = (double*) malloc(nelements*sizeof(double));

    // Fill it with zeroes:
    for(j=0; j < nelements; j++){

        Vector[j]=0.0;

    }

    // Return it:
    return Vector;
}

/*
double** MakeArray(int rows, int columns){

    // Initialize variables:
    int i,j; 
    double** theArray;

    // Allocate memory:
    theArray = (double**) malloc(rows*sizeof(double*));

    for(i=0;i<rows;i++){

        theArray[i] = (double*) malloc(columns*sizeof(double));

    }

    //Fill the array with zeroes (i.e. we clean it)
    for(i=0;i<rows;i++){

        for(j=0;j<columns;j++){

            theArray[i][j]=0.0;

        }

    }

    return theArray;
}
*/

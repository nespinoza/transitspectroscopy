#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#define ARRAYD(p) ((double *) (((PyArrayObject *)p)->data)) 

/* 
 *                                [INITIALIZATION]
 * ------------------ PROTOTYPES FOR FUNCTIONS AND EXTERNAL VARIABLES -----------------------
 *
 */

double* MakeVector(int nelements);                                   /* Function that makes/allocates a pointer in memory     */


/*                  [INITIALIZATION OF A METHOD]
*------------------------THE OBTAINP METHOD-----------------------------
* PyObject initialization: We define a PyObject which defines a Method 
* for the Marsh module: The ObtainP method which returns the P[i][j] spatial
* light fractions back to Python. BE CAREFUL WITH THE INDEXES, remember
* that i=rows, j=columns. According to Marsh's (1989) variables, j=X and
* i=Y.
*----------------------------------------------------------------------
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
 * input_xdata     : Vector containing the x-axis of the input data.
 * 
 * input_ydata     : Vector containing the input data.
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
    g_exp_constant = 1. / ( 2. * pow(sigma,2) );

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

static PyMethodDef CCFMethods[] = {
	{"Gaussian", CCF_Gaussian, METH_VARARGS, "Function that performs simple CCF of an input array with a gaussian."},
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

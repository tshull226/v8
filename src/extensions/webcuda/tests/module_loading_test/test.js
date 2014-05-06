var DEBUG = 1;

load("tests/Profiler/Profiler.js");

var profiler = new Profiler();
var module;

var dev = webcuda.Device(0);
var ctx = webcuda.Context(0, dev);

profiler.start("bs_cu");
module = webcuda.compileFile("tests/blackscholes/WebCUDA/bs_kernel");
profiler.stop("bs_cu");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("bs_ptx");
module = webcuda.moduleLoad("tests/blackscholes/WebCUDA/ptx/bs_kernel.ptx");
profiler.stop("bs_ptx");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("histogram_cu");
module = webcuda.compileFile("tests/histogram/WebCUDA/histogram_kernel");
profiler.stop("histogram_cu");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("histogram_ptx");
module = webcuda.moduleLoad("tests/histogram/WebCUDA/ptx/histogram_kernel.ptx");
profiler.stop("histogram_ptx");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("nbody_cu");
module = webcuda.compileFile("tests/nbody/WebCUDA/nbody");
profiler.stop("nbody_cu");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("nbody_ptx");
module = webcuda.moduleLoad("tests/nbody/WebCUDA/ptx/nbody.ptx");
profiler.stop("nbody_ptx");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("convolution_cu");
module = webcuda.compileFile("tests/image_convolution/WebCUDA/convolutionSeparable");
profiler.stop("convolution_cu");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("convolution_ptx");
module = webcuda.moduleLoad("tests/image_convolution/WebCUDA/ptx/convolutionSeparable.ptx");
profiler.stop("convolution_ptx");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("random_cu");
module = webcuda.compileFile("tests/random_pixel/WebCUDA/random");
profiler.stop("random_cu");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

profiler.start("random_ptx");
module = webcuda.moduleLoad("tests/random_pixel/WebCUDA/ptx/random.ptx");
profiler.stop("random_ptx");
if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
webcuda.moduleUnload(module);

webcuda.destroyCtx(ctx);

profiler.print();

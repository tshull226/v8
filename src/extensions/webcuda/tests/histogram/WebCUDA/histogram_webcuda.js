DEBUG = 0;

function histogram_webcuda(h_Data, byteCount, profiler) {
	//Retrieving Device
	if(DEBUG) print("retrieving Device Info");
	profiler.start("Retrieving device info");
	var dev = webcuda.Device(0);
	profiler.stop("Retrieving device info");

	//Setting up Context for CUDA Device
	if(DEBUG) print("creating Context");
	profiler.start("Creating CUDA context");
	var ctx = webcuda.Context(0, dev);
	profiler.stop("Creating CUDA context");

	//Creating host memory for pixel array
	if(DEBUG) print("creating host memory");
	profiler.start("Allocating host memory");
	var h_Histogram = new Uint32Array(256); 
	//var h_PartialHistograms = new Uint32Array(240 * 256); 
	profiler.stop("Allocating host memory");

	//Creating device memory for pixel array
	if(DEBUG) print("allocating CUDA memory");
	profiler.start("Allocating CUDA memory");
	var d_Data = webcuda.memAlloc(h_Data.buffer.byteLength);
	var d_Histogram = webcuda.memAlloc(h_Histogram.buffer.byteLength);
	var d_PartialHistograms = webcuda.memAlloc(240 * h_Histogram.buffer.byteLength);
	//if(DEBUG) print("size: "+d_I.size+" error: "+d_I.error);
	profiler.stop("Allocating CUDA memory");

	//Sending Input Data to CUDA Device Memory
	if(DEBUG) print("copying input data to device");
	profiler.start("copyHtoD");
	webcuda.copyHtoD(d_Data, h_Data.buffer);
	profiler.stop("copyHtoD");

	//Loading Module
	if(DEBUG) print("loading CUDA module");
	profiler.start("Loading CUDA module");
	//var module = webcuda.moduleLoad("tests/histogram/WebCUDA/histogram_kernel.ptx");
	var module = webcuda.compileFile("tests/histogram/WebCUDA/histogram_kernel");
	if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
	profiler.stop("Loading CUDA module");

	//Retrieving Function from Module
	if(DEBUG) print("retrieving function from module");
	profiler.start("Retrieving function from module");
	var cuFunc1 = webcuda.getFunction(module, "histogram256Kernel");
	if(DEBUG) print("name: " + cuFunc1.name + " error: " + cuFunc1.error);
	var cuFunc2 = webcuda.getFunction(module, "mergeHistogram256Kernel");
	if(DEBUG) print("name: " + cuFunc2.name + " error: " + cuFunc2.error);
	profiler.stop("Retrieving function from module");

	//Launching the Kernel
	if(DEBUG) print("trying to launch kernel");
	profiler.start("kernel");
	var launchResult = webcuda.launchKernel(cuFunc1, [240,1,1], [192,1,1], 0,
						[ {"memParam" : d_PartialHistograms}, {"memParam" : d_Data},
						  {"intParam" : byteCount / 4}
						]);
	if(DEBUG) print("launch result: " + launchResult);
        //webcuda.synchronizeCtx();
	//webcuda.copyDtoH(h_PartialHistograms.buffer, d_PartialHistograms);
        /*for(var i = 0; i < 240 * 256; i++) {
            if(DEBUG) print(h_PartialHistograms[i]);
        }*/
	var launchResult = webcuda.launchKernel(cuFunc2, [256,1,1], [256,1,1], 0,
						[ {"memParam" : d_Histogram}, {"memParam" : d_PartialHistograms},
						  {"intParam" : 240}
						]);
	if(DEBUG) print("launch result: " + launchResult);
        webcuda.synchronizeCtx();
	profiler.stop("kernel");

	//Retrieving Data from CUDA Device Memory
	if(DEBUG) print("copying CUDA Mem Result to host");
	profiler.start("copyDtoH");
	webcuda.copyDtoH(h_Histogram.buffer, d_Histogram);
	profiler.stop("copyDtoH");

	/*
	//temp check to see if things seem reasonable
	if(DEBUG) print("checking results");
	for(i = 0; i < numPixels; i++){
	if(DEBUG) print(h_I[i]);
	}
	*/

	//Freeing CUDA Memory
	if(DEBUG) print("freeing CUDA memory");
	profiler.start("Freeing CUDA memory");
	var memFree = webcuda.free(d_Histogram);
	if(DEBUG) print("free memory result: " + memFree);
	var memFree = webcuda.free(d_PartialHistograms);
	if(DEBUG) print("free memory result: " + memFree);
	var memFree = webcuda.free(d_Data);
	if(DEBUG) print("free memory result: " + memFree);
	profiler.stop("Freeing CUDA memory");

	//Freeing CUDA Module
	if(DEBUG) print("freeing CUDA module");
	profiler.start("Freeing CUDA module");
	var moduleFree = webcuda.moduleUnload(module);
	if(DEBUG) print("free module result: " + moduleFree);
	profiler.stop("Freeing CUDA module");

	//Destroying CUDA context
	if(DEBUG) print("destroying CUDA context");
	profiler.start("Destorying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	if(DEBUG) print("free context result: "+ ctxFree);
	profiler.stop("Destorying CUDA context");

	//returning value
	return h_Histogram;
}

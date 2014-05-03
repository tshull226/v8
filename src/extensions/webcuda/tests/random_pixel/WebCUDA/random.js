var DEBUG = 1;

INT_SIZE = 4;

//setting up pixel dimensions
width = 640;
height = 480;
numElements = height * width;
numPixels = 4 * numElements;

function main(){
	var seed = 1;

	load("tests/Profiler/Profiler.js");
	var profiler = new Profiler();

	webcuda.startProfiling();
	profiler.start("Total");
	var h_I = runCuda(seed, profiler);
	profiler.stop("Total");
	profiler.print();
	webcuda.stopProfiling();

	//temp check to see if things seem reasonable
	/*
		 for(i = 0; i < numPixels; i++){
		 if(DEBUG) print(t_I[i]);
		 }
		 */

	var t_I = runJS(seed);
	testResult(t_I, h_I);
}

function runCuda(seed, profiler){
	//Retrieving Device
	if(DEBUG) print("retrieving Device Info");
	profiler.start("Retrieving device info");
	var dev = webcuda.Device(0);
	profiler.stop("Retrieving device info");

	//Setting up Context for CUDA Device
	if(DEBUG) print("creating CUDA context");
	profiler.start("Creating CUDA context");
	var ctx = webcuda.Context(0, dev);
	profiler.stop("Creating CUDA context");

	//Creating host memory for pixel array
	if(DEBUG) print("creating host memory");
	profiler.start("Allocating host memory");
	var h_I = new Int32Array(numPixels); 
	profiler.stop("Allocating host memory");

	//Creating device memory for pixel array
	if(DEBUG) print("allocating CUDA memory");
	profiler.start("Allocating CUDA memory");
	var d_I = webcuda.memAlloc(h_I.buffer.byteLength);
	if(DEBUG) print("size: "+d_I.size+" error: "+d_I.error);
	profiler.stop("Allocating CUDA memory");

	//Loading Module
	if(DEBUG) print("loading CUDA module");
	profiler.start("Loading CUDA module");
	var module = webcuda.moduleLoad("tests/random_pixel/WebCUDA/random.ptx");
	if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
	profiler.stop("Loading CUDA module");

	//Retrieving Function from Module
	if(DEBUG) print("retrieving function from module");
	profiler.start("Retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "rng");
	if(DEBUG) print("name: " + cuFunc.name + " error: " + cuFunc.error);
	profiler.stop("Retrieving function from module");

	//Launching the Kernel
	if(DEBUG) print("trying to launch kernel");
	profiler.start("kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [40,30,1], [16,16,1], [{"memParam" : d_I}, {"intParam" : 1} ]);
	if(DEBUG) print("launch result: " + launchResult);
	profiler.stop("kernel");

	//Retrieving Data from CUDA Device Memory
	if(DEBUG) print("copying CUDA Mem Result to device");
	profiler.start("copyDtoH");
	webcuda.copyDtoH(h_I.buffer, d_I);
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
	var memFree = webcuda.free(d_I);
	if(DEBUG) print("free memory result: "+memFree);
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
	return h_I;

}

function runJS(seed){
	//creating array for testing in javascript to verify results
	var t_I = new Int32Array(numPixels);

	var delta = 0x9E3779B9;
	var k0 = 0xA341316C;
	var k1 = 0xC8013EA4;
	var k2 = 0xAD90777D;
	var k3 = 0x7E95761E;
	var ITER = 15;

	var i,j;
	for(i = 0; i < numElements; i++){

		var x = seed;
		var y = seed << 3;

		x += i + (i << 11) + (i << 19);
		y += i + (i << 9) + (i << 21);    

		var sum = 0;
		for (j=0; j < ITER; j++) {
			sum += delta;
			x += ((y << 4) + k0) & (y + sum) & ((y >> 5) + k1);
			y += ((x << 4) + k2) & (x + sum) & ((x >> 5) + k3);
		}

		var r = x & 0xFF;
		var g = (x & 0xFF00) >> 8;

		t_I[i*4  ] = r;
		t_I[i*4+1] = r;
		t_I[i*4+2] = r;
		t_I[i*4+3] = g;

	}

	return t_I;

}

function testResult(h_I, t_I){
	var i;
	for (i = 0; i < numElements*4; i++)
	{
		if (Math.abs(t_I[i] - h_I[i]) > 1e-5)
		{
			if(DEBUG) print("FAILED");
			if(DEBUG) print("Result verification failed at element " + i);
			if(DEBUG) print("Host element value: " + h_I[i] + ", CUDA element value: " + t_I[i]);
			quit();
		}
	}
	if(DEBUG) print("Test PASSED\n");
}

main();

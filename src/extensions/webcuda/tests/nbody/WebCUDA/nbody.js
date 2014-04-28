
INT_SIZE = 4;

var numBodies = 3;
var timeStep = .3;
var numIterations = 4;

function loadData(path){
	var data = read(path);
	var result = data.replace(/\r?\n|\r/g, " ");
	var temp = result.replace(/\s+/g,' ').trim();
	data = temp.split(' ');
	var length = data.length;
	var i;
	
	/*
	for(i = 0; i < length; i++){
		print(data[i]);
	}
	*/
	
	var j;
	numBodies = length/7;
	print("numbodies " + numBodies);

	var position = new Float32Array(numBodies*4);
	var acceleration = new Float32Array(numBodies*3); 

	for(i = 0; i < numBodies; i++){
		position[i*4 + 3] = parseFloat(data[i*7]);
		for(j = 0; j < 3; j++){
			position[i*4 + j] = parseFloat(data[i*7 + 1 + j]);
			acceleration[i*3 + j] = parseFloat(data[i*7 + 4 + j]);
		}
	}
	print(position);
	for(i = 0; i < numBodies; i++){
		print(position[i*4 + 3]);
	}
	print("\n\n\n****separator***\n\n\n");
	print(acceleration);

	return { "X" : position, "A" : acceleration};

}

function main(){


	var path = "tests/nbody/data/tab128";
	loadData(path);

	/*
		 var jsResult = runJS();
		 webcuda.startProfiling();
		 var cudaResult = runCuda();
		 webcuda.stopProfiling();
		 */

	//temp check to see if things seem reasonable
	/*
		 for(i = 0; i < numPixels; i++){
		 print(t_I[i]);
		 }
		 */

	//	testResult(jsResult, cudaResult);

}

function runCuda(){
	//Retrieving Device
	print("retrieving Device Info");
	var dev = webcuda.Device(0);

	//Setting up Context for CUDA Device
	print("creating Context");
	var ctx = webcuda.Context(0, dev);

	//Creating host memory for pixel array
	print("creating host memory");
	//
	var data = loadData();
	var h_X = data.X;
	var h_A = data.A;

	//Creating device memory for pixel array
	print("allocating CUDA memory");
	var d_X = webcuda.memAlloc(h_X.buffer.byteLength);
	var d_A = webcuda.memAlloc(h_A.buffer.byteLength);
	print("d_X size: "+d_X.size+" error: "+d_X.error);
	print("d_A size: "+d_A.size+" error: "+d_A.error);

	//Loading Module
	print("loading CUDA module");
	var module = webcuda.moduleLoad("tests/nbody/WebCUDA/nbody.ptx");
	print("fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "calculate_forces");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);


	//Calculating Grid and Block Sizes
	var blockSize = 5;
	var gridSize = numBodies/blockSize;
	//Calculating the number of shared memory bytes needed
	var sharedMem = 44;
	//Launching the Kernel
	print("trying to launch kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [blockSize,1,1], [gridSize,1,1], [{"sharedMemBytes" : sharedMem}, {"memParam" : d_X}, {"memParam" : d_A}, {"floatParam" : timeStep}, {"intParam" : numIterations}]);
	print("launch result: " + launchResult);

	//Synchronizing for Context to Complete
	webcuda.synchronizeCtx();

	//Retrieving Data from CUDA Device Memory
	print("copying CUDA Mem Result to device");
	//TODO should allow them to be asynchronous
	h_X = new Float32Array(numBodies); 
	h_A = new Float32Array(numBodies); 
	webcuda.copyDtoH(h_X.buffer, d_X);
	webcuda.copyDtoH(h_A.buffer, d_A);

	/*
	//temp check to see if things seem reasonable
	print("checking results");
	for(i = 0; i < numPixels; i++){
	print(h_I[i]);
	}
	*/

	//Freeing CUDA Memory
	print("freeing CUDA memory");
	var memFree = webcuda.free(d_X);
	print("d_X free memory result: "+memFree);
	var memFree = webcuda.free(d_A);
	print("d_A free memory result: "+memFree);

	//Freeing CUDA Module
	print("freeing CUDA module");
	var moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);

	//Destroying CUDA context
	print("destroying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free context result: "+ ctxFree);

	//returning value
	return {"positions" : d_X, "accelerations" : d_A};

}

function runJS(){
	//TODO
}

function testResult(jsResult, cudaResult){
	var i;
	var jsPosition = jsResult.positions;
	var cudaPosition = cudaResult.positions;
	for (i = 0; i < numBodies*3; i++)
	{
		if (Math.abs(jsPosition[i] - cudaPosition[i]) > 1e-5)
		{
			print("FAILED");
			print("Result verification failed at position element " + Math.floor(i/3) + " coordinate " + i%3);
			print("CUDA element value: " + cudaPosition[i] + ", JavaScript element value: " + jsPosition[i]);
			quit();
		}
	}

	var jsAcceleration = jsResult.accelerations;
	var cudaAcceleration = cudaResult.accelerations;
	for (i = 0; i < numBodies; i++)
	{
		if (Math.abs(jsAcceleration[i] - cudaAcceleration[i]) > 1e-5)
		{
			print("FAILED");
			print("Result verification failed at acceleration element " + i);
			print("CUDA element value: " + cudaAcceleration[i] + ", JavaScript element value: " + jsAcceleration[i]);
			quit();
		}
	}
	print("Test PASSED\n");
}

main();

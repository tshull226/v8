
INT_SIZE = 4;

	//setting up pixel dimensions
	width = 640;
	height = 480;
	numElements = height * width;
	numPixels = 4 * numElements;

function main(){
	var seed = 1;


	var t_I = runJS(seed);
	var h_I = runCuda(seed);

	//temp check to see if things seem reasonable
	/*
	for(i = 0; i < numPixels; i++){
		print(t_I[i]);
	}
	*/

	testResult(t_I, h_I);
}

function runCuda(seed){
	var blocks, threads;

	//Creating host memory for pixel array
	print("creating host memory");
	var h_I = new Int32Array(numPixels); 

	//Retrieving Device
	print("retrieving Device Info");
	var dev = webcuda.Device(0);

	//Setting up Context for CUDA
	print("creating Context");
	var ctx = webcuda.newContext(0, dev);

	//Creating device memory for pixel array
	print("allocating CUDA memory");
	//print(h_I.buffer.byteLength);
	var d_I = webcuda.memAlloc(h_I.buffer.byteLength);
	print("size: "+d_I.size+" error: "+d_I.error);


	
	//Loading Module
	print("loading CUDA module");
	var module = webcuda.moduleLoad("tests/random_pixel/WebCUDA/random.ptx");
	print("fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "rng");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);

	//Launching the Kernel
	
	print("trying to launch kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [40,30,1], [16,16,1], [{"memParam" : d_I}, {"intParam" : 1} ]);
	print("launch result: " + launchResult);
	
	
	//Retrieving Data from CUDA Device Memory

	print("copying CUDA Mem Result to device");
	webcuda.copyDtoH(h_I.buffer, d_I);

	/*
	//temp check to see if things seem reasonable
	print("checking results");
	var value = 0;
	for(i = 0; i < numPixels; i++){
		//value += h_I[i];
		print(h_I[i]);
	}
	*/



	//Freeing CUDA Memory
	print("freeing CUDA memory");
	var memFree = webcuda.free(d_I);
	print("free memory result: "+memFree);

	//Freeing CUDA context
	var ctxFree = ctx.destroy();
	print("free context result: "+ ctxFree);
	
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
			print("FAILED");
			print("Result verification failed at element " + i);
			print("Host element value: " + h_I[i] + ", CUDA element value: " + t_I[i]);
			quit();
		}
	}
	print("Test PASSED\n");
}

main();

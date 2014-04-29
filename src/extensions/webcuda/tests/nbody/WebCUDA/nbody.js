
INT_SIZE = 4;

var numBodies;
var timeStep = 0.01;
var numIterations = 4;
var threadSize = 16;
var blockSize = 8;

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
	var velocity = new Float32Array(numBodies*3); 

	for(i = 0; i < numBodies; i++){
		position[i*4 + 3] = parseFloat(data[i*7]);
		for(j = 0; j < 3; j++){
			position[i*4 + j] = parseFloat(data[i*7 + 1 + j]);
			velocity[i*3 + j] = parseFloat(data[i*7 + 4 + j]);
		}
	}
	print(position);
	for(i = 0; i < numBodies; i++){
		print(position[i*4 + 3]);
	}
	print("\n\n\n****separator***\n\n\n");
	print(velocity);

	return { "X" : position, "V" : velocity};

}

function loadDataJS(path){
	var data = read(path);
	var result = data.replace(/\r?\n|\r/g, " ");
	var temp = result.replace(/\s+/g,' ').trim();
	data = temp.split(' ');
	var length = data.length;
	var i;

	numBodies = length/7;

	result = new Array();

	for(i = 0; i < numBodies; i++ ){
		result[i] = new Body(data, i*7);
	}

	return result;
}

function main(){


	var path = "tests/nbody/data/tab128";
//	loadData(path);

	runJS(path);
	runCuda(path);
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

function runCuda(path){
	//Retrieving Device
	print("retrieving Device Info");
	var dev = webcuda.Device(0);

	//Setting up Context for CUDA Device
	print("creating Context");
	var ctx = webcuda.Context(0, dev);

	//Creating host memory for pixel array
	print("creating host memory");
	//
	var data = loadData(path);
	var h_X = data.X;
	var h_V = data.V;

	//Creating device memory for pixel array
	print("allocating CUDA memory");
	var d_X = webcuda.memAlloc(h_X.buffer.byteLength);
	var d_V = webcuda.memAlloc(h_V.buffer.byteLength);
	print("d_X size: "+d_X.size+" error: "+d_X.error);
	print("d_A size: "+d_V.size+" error: "+d_V.error);

	//copying data to device
	print("copying CUDA initial parameters to device");
	//TODO should allow them to be asynchronous
	webcuda.copyHtoD(d_X, h_X.buffer);
	webcuda.copyHtoD(d_V, h_V.buffer);

	//Loading Module
	print("compiling CUDA module");
	var module = webcuda.compileFile("tests/nbody/WebCUDA/nbody");
	//TODO fix this
	print("cuName: " + module.cuName + " fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "calculate_forces");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);


	//Calculating the number of shared memory bytes needed
	var sharedMem = threadSize*4*4;

	
	//Launching the Kernel
	print("trying to launch kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [blockSize,1,1], [threadSize,1,1], sharedMem, [{"memParam" : d_X}, {"memParam" : d_V},{"intParam" : numBodies} , {"intParam" : numIterations}, {"floatParam" : timeStep}]);
	print("launch result: " + launchResult);
	print("launched kernel...");

	//Synchronizing for Context to Complete
	webcuda.synchronizeCtx();
	

	//Retrieving Data from CUDA Device Memory
	print("copying CUDA Mem Result to device");
	//TODO should allow them to be asynchronous
	h_X = new Float32Array(numBodies); 
	var copyDtoH = webcuda.copyDtoH(h_X.buffer, d_X);
	print("copying result: " + copyDtoH);

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
	var memFree = webcuda.free(d_V);
	print("d_V free memory result: "+memFree);

	//Freeing CUDA Module
	//print("freeing CUDA module");
	//var moduleFree = webcuda.moduleUnload(module);
	//print("free module result: " + moduleFree);

	//Destroying CUDA context
	print("destroying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free context result: "+ ctxFree);

	//returning value
	return {"positions" : h_X};

}

function runJS(path){
	//TODO
	var bodies = loadDataJS(path);

	for(var i = 0; i < numIterations; i++){
		advance(bodies, timeStep);
	}

	return bodies;
}

function Body(data, index){
	this.mass = data[index];
	this.x = data[index + 1];
	this.y = data[index + 2];
	this.z = data[index + 3];

	this.vx = data[index + 4];
	this.vy = data[index + 5];
	this.vz = data[index + 6];
}

function advance(bodies, step){
	var dx, dy, dz, distance, mag, dt;
	var i,j;
	var bodyi, bodyj, body;

	dt = step;
	for(i = 0; i < numBodies; i++){

		bodyi = bodies[i];
		for (j=i+1; j<numBodies; j++) {
			bodyj = bodies[j];
			dx = bodyi.x - bodyj.x;
			dy = bodyi.y - bodyj.y;
			dz = bodyi.z - bodyj.z;

			distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
			mag = dt / (distance * distance * distance);

			bodyi.vx -= dx * bodyj.mass * mag;
			bodyi.vy -= dy * bodyj.mass * mag;
			bodyi.vz -= dz * bodyj.mass * mag;

			bodyj.vx += dx * bodyi.mass * mag;
			bodyj.vy += dy * bodyi.mass * mag;
			bodyj.vz += dz * bodyi.mass * mag;
		}
	}

	for (i=0; i<numBodies; i++) {
		body = bodies[i];
		body.x += dt * body.vx;
		body.y += dt * body.vy;
		body.z += dt * body.vz;
	}
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

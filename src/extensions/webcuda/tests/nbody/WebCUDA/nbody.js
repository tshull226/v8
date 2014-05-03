var DEBUG = 0;

var numBodies;
var timeStep = 0.01;
var numIterations = 40;
var threadSize = 16;
var blockSize = 8;
var filepath = "tests/nbody/data/tab1024";

function loadData(path){
	var data = read(path);
	var result = data.replace(/\r?\n|\r/g, " ");
	var temp = result.replace(/\s+/g,' ').trim();
	data = temp.split(' ');
	var length = data.length;
	var i;

	var j;
	numBodies = length/7;
	if(DEBUG) print("numbodies " + numBodies);

	var position = new Float32Array(numBodies*4);
	var velocity = new Float32Array(numBodies*3); 

	for(i = 0; i < numBodies; i++){
		position[i*4 + 3] = parseFloat(data[i*7]);
		for(j = 0; j < 3; j++){
			position[i*4 + j] = parseFloat(data[i*7 + 1 + j]);
			velocity[i*3 + j] = parseFloat(data[i*7 + 4 + j]);
		}
	}

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
	load("tests/Profiler/Profiler.js");

	webcuda.startProfiling();
	var profiler = new Profiler();
	profiler.start("Total");
	var cudaResult = runCuda(filepath, profiler);
	profiler.stop("Total");
	profiler.print();
	webcuda.stopProfiling();

	//var jsResult = runJS(filepath);
	//testResult(jsResult, cudaResult);
}

function runCuda(path, profiler){
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
	var data = loadData(path);
	var h_X = data.X;
	var h_V = data.V;
	profiler.stop("Allocating host memory");

	//Creating device memory for pixel array
	if(DEBUG) print("allocating CUDA memory");
	profiler.start("Allocating CUDA memory");
	var d_X = webcuda.memAlloc(h_X.buffer.byteLength);
	var d_V = webcuda.memAlloc(h_V.buffer.byteLength);
	if(DEBUG) print("d_X size: "+d_X.size+" error: "+d_X.error);
	if(DEBUG) print("d_A size: "+d_V.size+" error: "+d_V.error);
	profiler.stop("Allocating CUDA memory");

	//copying data to device
	if(DEBUG) print("copying CUDA initial parameters to device");
	profiler.start("copyHtoD");
	//TODO should allow them to be asynchronous
	webcuda.copyHtoD(d_X, h_X.buffer);
	webcuda.copyHtoD(d_V, h_V.buffer);
	profiler.stop("copyHtoD");

	//Loading Module
	if(DEBUG) print("compiling CUDA module");
	profiler.start("Loading CUDA module");
	var module = webcuda.compileFile("tests/nbody/WebCUDA/nbody");
	//TODO fix this
	if(DEBUG) print("cuName: " + module.cuName + " fname: " + module.fname + " error: " + module.error);
	profiler.stop("Loading CUDA module");

	//Retrieving Function from Module
	if(DEBUG) print("retrieving function from module");
	profiler.start("Retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "calculate_forces");
	if(DEBUG) print("name: " + cuFunc.name + " error: " + cuFunc.error);
	profiler.stop("Retrieving function from module");


	//Calculating the number of shared memory bytes needed
	var sharedMem = threadSize*4*4;
	if(DEBUG) print("shared memory size: " + sharedMem);


	//Launching the Kernel
	if(DEBUG) print("trying to launch kernel");
	profiler.start("kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [blockSize,1,1], [threadSize,1,1], sharedMem, [{"memParam" : d_X}, {"memParam" : d_V},{"intParam" : numBodies} , {"intParam" : numIterations}, {"floatParam" : timeStep}]);
	if(DEBUG) print("launch result: " + launchResult);
	if(DEBUG) print("launched kernel...");
	profiler.stop("kernel");

	//Synchronizing for Context to Complete
	webcuda.synchronizeCtx();


	//Retrieving Data from CUDA Device Memory
	if(DEBUG) print("copying CUDA Mem Result to device");
	profiler.start("copyDtoH");
	//TODO should allow them to be asynchronous
	h_X = new Float32Array(numBodies); 
	var copyDtoH = webcuda.copyDtoH(h_X.buffer, d_X);
	if(DEBUG) print("copying result: " + copyDtoH);
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
	var memFree = webcuda.free(d_X);
	if(DEBUG) print("d_X free memory result: "+memFree);
	var memFree = webcuda.free(d_V);
	if(DEBUG) print("d_V free memory result: "+memFree);
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
	var jsPosition = new Array();
	var cudaPosition = cudaResult.positions;
	//making js result correct
	for(i = 0; i < numBodies; i++){
		jsPosition[i*4] = jsResult[i].x;
		jsPosition[i*4+1] = jsResult[i].y;
		jsPosition[i*4+2] = jsResult[i].z;
		jsPosition[i*4+3] = jsResult[i].mass;
	}

	for (i = 0; i < numBodies*4; i++)
	{
		if (Math.abs(jsPosition[i] - cudaPosition[i]) > 1e-5)
		{
			if(DEBUG) print("FAILED");
			if(DEBUG) print("Result verification failed at position element " + Math.floor(i/4) + " coordinate " + i%4);
			if(DEBUG) print("CUDA element value: " + cudaPosition[i] + ", JavaScript element value: " + jsPosition[i]);
			quit();
		}
	}

	if(DEBUG) print("PASSED");
}

main();

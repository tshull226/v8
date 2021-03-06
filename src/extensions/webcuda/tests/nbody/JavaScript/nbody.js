var numBodies;
var timeStep = 0.01;
var numIterations = 10;
var threadSize = 16;
var blockSize = 8;
var filepath = "tests/nbody/data/tab1024";
//var filepath = "tests/nbody/data/tab8096";


function loadDataJS(path, profiler) {
	var data = read(path);
	var result = data.replace(/\r?\n|\r/g, " ");
	var temp = result.replace(/\s+/g,' ').trim();
	data = temp.split(' ');
	var length = data.length;
	var i;

	numBodies = length/7;

	profiler.start("Allocating host memory");
	result = new Array();
	for(i = 0; i < numBodies; i++ ){
		result[i] = new Body(data, i*7);
	}
	profiler.stop("Allocating host memory");

	return result;
}

function main(){
	load("tests/Profiler/Profiler.js");
	print("STARTING");
	var profiler = new Profiler();
	//profiler.start("Total");
	var jsResult = runJS(filepath, profiler);
	//profiler.stop("Total");
	profiler.print();
	print("DONE");
}


function runJS(path, profiler){
	//TODO
	//profiler.start("Allocating host memory");
	var bodies = loadDataJS(path, profiler);
	//profiler.stop("Allocating host memory");

	profiler.start("Computation");
	for(var i = 0; i < numIterations; i++){
		advance(bodies, timeStep);
	}
	profiler.stop("Computation");

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

main();

function Profiler() {
  var start = {};
  var elasped_time = {};

  %BeginTiming();

  function start_timer(name) {
    if(name == undefined) print("[ERROR] Profiler.start() needs a timer name.");
    if(name in start) {
      print("[WARNING] Already started: " + name);
    }
    start[name] = %GetTiming();
  }

  function stop_timer(name) {
    if(name == undefined) print("[ERROR] Profiler.stop() needs a timer name.");
    if(name in start) {
      if(name in elasped_time) {
        elasped_time[name] += %GetTiming() - start[name];
      } else {
        elasped_time[name] = %GetTiming() - start[name];
      }
      delete start[name];
    } else {
      print("[ERROR] Not started: " + name);
    }
  }

  function print_result() {
    for(key in elasped_time) {
      if("Total" in elasped_time) {
        //print(key + ": " + String(elasped_time[key] / elasped_time["Total"] * 100) + "%");
        //print(key + ": " + String(Math.round(elasped_time[key])) + " (" + String(Math.round(elasped_time[key] / elasped_time["Total"] * 10000) / 100) + "%)");
        print(key + ", " + String(Math.round(elasped_time[key])));
      } else {
        print(key + ", " + String(Math.round(elasped_time[key])));
      }
    }
  }

  this.start = function(name) {
    start_timer(name);
  }

  this.stop = function(name) {
    stop_timer(name);
  }

  this.print = function() {
    print_result();
  }
}

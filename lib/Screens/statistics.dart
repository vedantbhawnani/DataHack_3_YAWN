import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:syncfusion_flutter_charts/sparkcharts.dart';


class Statistics extends StatefulWidget {
  const Statistics({super.key});

  @override
  State<Statistics> createState() => _StatisticsState();
}

class _StatisticsState extends State<Statistics> {
  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            height: mediaQuery.size.height*0.5,
            width: mediaQuery.size.width*0.8,
            child: Padding(
              padding: EdgeInsets.all(mediaQuery.size.height*0.04),
              child: SfCartesianChart(
                title: ChartTitle(
                  text: 'Weekly Workout',
                  textStyle: TextStyle(
                    color: Colors.black,
                    fontFamily: 'Vercitti',
                    fontSize: mediaQuery.size.height*0.02,
                  ),
                ),
                primaryXAxis: CategoryAxis(),
                  series: <ChartSeries<WeeklyWorkout, String>>[
                    BarSeries<WeeklyWorkout, String>(
                      // Bind data source
                        dataSource:  <WeeklyWorkout>[
                          WeeklyWorkout('Mon', 35),
                          WeeklyWorkout('Tue', 28),
                          WeeklyWorkout('Wed', 34),
                          WeeklyWorkout('Thurs', 32),
                          WeeklyWorkout('Fri', 40)
                        ],
                        xValueMapper: (WeeklyWorkout weekday, _) => weekday.weekday,
                        yValueMapper: (WeeklyWorkout weekday, _) => weekday.time,
                      yAxisName: 'Weekdays',
                      xAxisName: 'Workout (in minutes)',
                    )
                  ]

              )
            ),
          ),
          Container(
            height: mediaQuery.size.height*0.5,
            width: mediaQuery.size.width*0.8,
            child: SfCircularChart(
              legend: Legend(
                isVisible: true,
              ),
              title: ChartTitle(
                text:'Workout Reps',
                textStyle: TextStyle(
                  color: Colors.black,
                  fontFamily: 'Vercitti',
                  fontSize: mediaQuery.size.height*0.02,
                ),
              ),
              series: <CircularSeries>[
                RadialBarSeries<WorkoutReps,String>(
                  dataSource: <WorkoutReps>[
                      WorkoutReps('Push Ups', 20),
                      WorkoutReps('Squats', 10),
                      WorkoutReps('Overhead Press', 15),
                  ],
                  xValueMapper: (WorkoutReps workout, _) => workout.workout,
                  yValueMapper: (WorkoutReps workout, _) => workout.reps,
                  dataLabelSettings: DataLabelSettings(isVisible: true,),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class WeeklyWorkout{
  final String weekday;
  final int time;
  WeeklyWorkout(this.weekday, this.time);
}

class WorkoutReps{
  final String workout;
  final int reps;
  WorkoutReps(this.workout,this.reps);
}

import 'package:exerciserecognition/constants.dart';
import 'package:flutter/material.dart';

class WorkoutPlanPage extends StatefulWidget {
  final double bmi;

  const WorkoutPlanPage({Key? key, required this.bmi}) : super(key: key);

  @override
  _WorkoutPlanPageState createState() => _WorkoutPlanPageState();
}

class _WorkoutPlanPageState extends State<WorkoutPlanPage> {
  final List<Difficulty> _difficulties = [
    Difficulty.EASY,
    Difficulty.MEDIUM,
    Difficulty.HARD,
  ];

  final List<List<Exercise>> _exercisesByDifficulty = [
    // EASY exercises
    [
      Exercise(
          name: 'Push-ups',
          difficulty: Difficulty.EASY,
          reps: 10
      ),
      Exercise(
          name: 'Squats',
          difficulty: Difficulty.EASY,
          reps: 15),
      Exercise(
          name: 'Crunches',
          difficulty: Difficulty.EASY,
          reps: 20),
    ],
    // MEDIUM exercises
    [
      Exercise(
          name: 'Lunges',
          difficulty: Difficulty.MEDIUM,
          reps: 10),
      Exercise(
          name: 'Plank',
          difficulty: Difficulty.MEDIUM,
          reps: 30
      ),
      Exercise(
          name: 'Pull-ups',
          difficulty: Difficulty.MEDIUM,
          reps: 5
      ),
    ],
    // HARD exercises
    [
      Exercise(
          name: 'Burpees',
          difficulty: Difficulty.HARD,
          reps: 10
      ),
      Exercise(
          name: 'Mountain climbers',
          difficulty: Difficulty.HARD,
          reps: 15
      ),
      Exercise(
          name: 'Russian twists',
          difficulty: Difficulty.HARD,
          reps: 20
      ),
    ],
  ];

  Difficulty getDifficultyBasedOnBMI() {
    print(widget.bmi);
    if (widget.bmi < 18.5) {
      return Difficulty.EASY;
    } else if (widget.bmi >= 18.5 && widget.bmi < 25) {
      return Difficulty.MEDIUM;
    } else {
      return Difficulty.HARD;
    }

  }

  @override
  Widget build(BuildContext context) {
    final difficulty = getDifficultyBasedOnBMI();
    final mediaQuery = MediaQuery.of(context);
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Container showing the difficulty
            Text(
              difficulty.name,
              style: TextStyle(
                color: Colors.black,
                fontSize: mediaQuery.size.height*0.03,
                fontFamily: 'ObjectSans',
              ),
            ),
            // Workout plan
            Container(
              color: Colors.white,
              width: mediaQuery.size.width*0.4,
              child: ListView.builder(
                shrinkWrap: true,
                itemCount: _exercisesByDifficulty[difficulty.index].length,
                itemBuilder: (context, index) {
                  final exercise = _exercisesByDifficulty[difficulty.index][index];
                  return ListTile(
                    title: Text(
                        exercise.name,
                        style: TextStyle(
                          color: kBlackBackgroundColor,
                          fontSize: mediaQuery.size.height*0.03,
                          fontFamily: 'Vercetti',
                        ),
                    ),
                    subtitle: Text('${exercise.reps} reps', style: TextStyle(
                      fontFamily: 'serif',
                      fontSize: mediaQuery.size.height*0.02,
                    ),),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

enum Difficulty {
  EASY,
  MEDIUM,
  HARD,
}

class Exercise {
  final String name;
  final Difficulty difficulty;
  final int reps;

  Exercise({required this.name, required this.difficulty, required this.reps});

}

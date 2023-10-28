import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';

class GoalBased extends StatelessWidget {
  final String goal;
  const GoalBased({required this.goal,super.key});

  @override
  Widget build(BuildContext context) {
    print(goal);
    final mediaQuery = MediaQuery.of(context);
    String text='';
    if(goal=='Stay Fit'){
        text= '''
        STAY FIT\n
        Workout Plan:\n
        \tDuration: 30-45 minutes, 3-5 times per week.\n
        Exercises:\n
        \tCardio (e.g., Running, Cycling, Jump Rope): 15-20 minutes\n
        \tBodyweight Exercises (e.g., Push-ups, Squats, Planks): 3 sets of 10-15 reps each\n
        \tYoga or Stretching: 10-15 minutes\n
        Meal Plan:\n
        \tBreakfast: Oatmeal with nuts and fruits\n
        \tLunch: Quinoa or Brown Rice with mixed vegetables and tofu\n
        \tSnack: Greek yogurt with berries\n
        \tDinner: Lentil soup with a side salad\n
      ''';
    }
    else if(goal=='Gain Weight'){
      text = '''
        GAIN WEIGHT\n
        Workout Plan:\n
        \tDuration: 45-60 minutes, 4-6 times per week.\n
        Exercises:\n
        \tStrength Training (e.g., Weightlifting, Resistance Bands): 4 sets of 8-12 reps each\n
        \tCompound Exercises (e.g., Squats, Deadlifts, Bench Press): 3 sets of 6-10 reps each\n
        \tRest between sets: 1-2 minutes\n
        Meal Plan:
        \tBreakfast: Banana and peanut butter smoothie with oats\n
        \tLunch: Quinoa or Brown Rice with black beans and avocado\n
        \tSnack: Mixed nuts and dried fruits\n
        \tDinner: Grilled tofu with sweet potatoes and steamed broccoli\n
      ''';
    }
    else if(goal=='Lose Weight'){
      text= '''
          LOSE WEIGHT\n
          Workout Plan:\n
          \tDuration: 30-45 minutes, 4-6 times per week.\n
          Exercises:\n
          \tHigh-Intensity Interval Training (HIIT): 20-30 minutes\n
          \tBodyweight Exercises (e.g., Lunges, Push-ups, Burpees): 3 sets of 12-15 reps each\n
          \tRest between exercises: 30-45 seconds\n
          Meal Plan:\n
          \tBreakfast: Spinach and mushroom omelette\n
          \tLunch: Mixed green salad with chickpeas and vinaigrette\n
          \tSnack: Carrot and cucumber sticks with hummus\n
          \tDinner: Grilled vegetables with a small portion of quinoa or brown rice\n
      ''';
    }
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Container(
          child: Text(text,style: TextStyle(
            fontSize: mediaQuery.size.height*0.02,
          ),),
        ),
      ),
    );
  }
}

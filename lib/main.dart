import 'dart:ui';
import 'package:exerciserecognition/Screens/homescreen.dart';
import 'package:exerciserecognition/Screens/statistics.dart';
import 'package:flutter/material.dart';
import 'Screens/splashScreen2.dart';
import 'Screens/StartScreen.dart';
import 'Screens/startWorkout.dart';
import 'Screens/SignUp.dart';
import 'Screens/AgeWeightHeight.dart';
import 'Screens/login.dart';
import 'Screens/endWorkout.dart';

void main() {
  runApp(const RoutesClass());
}

class RoutesClass extends StatelessWidget {
  const RoutesClass({super.key});


  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: 'splashscreen',
      routes: {
        'splashscreen': (context)=> SplashScreen(),
        'startscreen' : (context)=> StartScreen(),
        'signup' : (context)=>SignUp(),
        'homescreen': (context) => HomeScreen(),
        'login' : (context) => LogIn(),
        'startWorkout': (context)=> CameraApp(),
        'endWorkout': (context) => EndWorkout(),
        'statistics': (context) => Statistics(),
      },
    );
  }
}

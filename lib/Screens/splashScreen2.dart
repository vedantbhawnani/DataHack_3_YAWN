import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../firebase_options.dart';

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _rotationAnimation;

  void getUserState() async{
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
  }
  @override
  void initState() {
    super.initState();
    getUserState();
    timeDilation = 5;

    // Initialize the animation controller.
    _animationController = AnimationController(
      vsync: this,
      duration: Duration(seconds: 1),
    );

    // Create a rotation animation.
    _rotationAnimation = Tween<double>(
      begin: 0.0,
      end: 360.0,
    ).animate(_animationController)
      ..addListener(() {
        setState(() {});
      });

    // Start the animation.
    _animationController.repeat();

    // Start a timer to navigate to the next screen after 5 seconds.
    Timer(Duration(seconds: 5), () => Navigator.pushReplacementNamed(context, 'startscreen'));
  }

  @override
  void dispose() {
    // Dispose of the animation controller.
    _animationController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Transform.rotate(
              angle: _rotationAnimation.value * 3.1415926 / 180.0,
              child: SvgPicture.asset('images/dumbell-svgrepo-com.svg', height: 120, color: Colors.white,),
            ),
          ],
        ),
      ),
    );
  }
}

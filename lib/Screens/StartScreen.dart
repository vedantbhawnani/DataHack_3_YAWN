import 'package:exerciserecognition/constants.dart';
import 'package:flutter/material.dart';


class StartScreen extends StatefulWidget {
  const StartScreen({super.key});

  @override
  State<StartScreen> createState() => _StartScreenState();
}

class _StartScreenState extends State<StartScreen> {
  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return Scaffold(
      body: Stack(
        children: [
          Opacity(
            opacity: 0.85,
            child: Container(
              decoration: const BoxDecoration(
                color: Colors.transparent,
                image: DecorationImage(
                  image: AssetImage('images/gymbro.jpg'),
                  fit: BoxFit.cover,
                ),
              ),
            ),
          ),
          Positioned(
            top: mediaQuery.size.height * 0.1,
            left: mediaQuery.size.width * 0.26,
            child: Row(
              children: [
                Text(
                  'FIT',
                  style: TextStyle(
                    fontFamily: 'ObjectSans',
                    fontSize: mediaQuery.size.height * 0.1,
                    color: Colors.white,
                  ),
                ),
                Text(
                  '.AI',
                  style: TextStyle(
                    fontFamily: 'ObjectSans',
                    fontSize: mediaQuery.size.height * 0.1,
                    color: kLightBlue,
                  ),
                ),
              ],
            ),
          ),
          Positioned(
            top: mediaQuery.size.height * 0.5,
            left: mediaQuery.size.width * 0.09,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Motivation',
                  style: TextStyle(
                    color: Colors.white,
                    fontFamily: 'ObjectSans',
                    fontSize: mediaQuery.size.height * 0.06,
                  ),
                ),
                Text(
                  'Accountability',
                  style: TextStyle(
                    color: Colors.white,
                    fontFamily: 'ObjectSans',
                    fontSize: mediaQuery.size.height * 0.06,
                  ),
                ),
                Row(
                  children: [
                    Text(
                      'Fit',
                      style: TextStyle(
                        color: kLightBlue,
                        fontFamily: 'ObjectSans',
                        fontSize: mediaQuery.size.height * 0.06,
                      ),
                    ),
                    Text(
                      'ness',
                      style: TextStyle(
                        color: Colors.white,
                        fontFamily: 'ObjectSans',
                        fontSize: mediaQuery.size.height * 0.06,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Positioned(
            top: mediaQuery.size.height * 0.68,
            left: mediaQuery.size.width * 0.09,
            child: Text(
              'Your Partner in Fitness Transformation \nand Goal Achievement',
              style: TextStyle(
                color: Colors.white,
                fontSize: mediaQuery.size.height*0.02,
              ),
            ),
          ),
          Positioned(
            top: mediaQuery.size.height * 0.75,
            left: mediaQuery.size.width * 0.1,
            child: TextButton(
              style: ButtonStyle(
                backgroundColor: MaterialStatePropertyAll(kLightBlue),
                shape: MaterialStatePropertyAll(
                  RoundedRectangleBorder(
                    borderRadius:
                        BorderRadius.circular(mediaQuery.size.height * 0.04),
                  ),
                ),
              ),
              child: Padding(
                padding: EdgeInsets.only(
                    top: mediaQuery.size.height * 0.02,
                    bottom: mediaQuery.size.height * 0.02,
                    left: mediaQuery.size.width * 0.3,
                    right: mediaQuery.size.width * 0.3),
                child: Text(
                  'Get Started',
                  style: TextStyle(
                    fontFamily: 'Vercetti',
                    color: Colors.white,
                    fontSize: mediaQuery.size.height * 0.02,
                  ),
                ),
              ),
              onPressed: () {
                  Navigator.pushNamed((context),'login');
              },
            ),
          ),
          Positioned(
            top: mediaQuery.size.height * 0.85,
            left: mediaQuery.size.width * 0.25,
            child: Row(
              children: [
                Text(
                  'Dont have an account?',
                  style: TextStyle(
                    fontSize: mediaQuery.size.height * 0.02,
                    color: Colors.white,
                  ),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pushNamed(context, 'signup');
                  },
                  child: Text(
                    'Sign Up now',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: mediaQuery.size.height * 0.02,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

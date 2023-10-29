import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class EndWorkout extends StatefulWidget {
  const EndWorkout({super.key});

  @override
  State<EndWorkout> createState() => _EndWorkoutState();
}

class _EndWorkoutState extends State<EndWorkout> {
  String pushUps = '30';
  String squats = '20';
  String overPress = '50';
  Map<String,String> request = {
    'Push Ups': '0',
  };
  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        children: [
          Container(
            padding: EdgeInsets.only(top: mediaQuery.size.height*0.05,left: mediaQuery.size.height*0.02,),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  'Yayy!!',
                  style: TextStyle(
                    color: kLightBlue,
                    fontSize: mediaQuery.size.height*0.04,
                  ),
                ),
                Text(
                  'You have successfully completed \nyour workout!!',
                  style: TextStyle(
                    color: kLightBlue,
                    fontSize: mediaQuery.size.height*0.04,
                  ),
                ),
                TextButton(
                  onPressed: (){
                    FirebaseFirestore db = FirebaseFirestore.instance;
                    FirebaseAuth auth = FirebaseAuth.instance;
                    DateTime uniqueId = DateTime.timestamp();
                    final String? userEmail = auth.currentUser?.email.toString();
                    String docName = '$userEmail $uniqueId';
                    request["Email"]=userEmail!;
                    request["Squats"]=squats;
                    request["OverHead Presses"]=overPress;
                    request["Push Ups"]=pushUps;
                    print(request);
                    db
                        .collection('Workout Details')
                        .doc(docName)
                        .set(request);
                    Navigator.pushNamed(context, 'homescreen');
                  },
                  child: Text('Save Workout Data'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

import 'package:exerciserecognition/Components/recommendationSystem.dart';
import 'package:exerciserecognition/Screens/goalBased.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:exerciserecognition/Service/auth.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  CollectionReference collectionRef =
      FirebaseFirestore.instance.collection('User Details');
  List dataFromDB = [];
  double bmi = 0.0;
  String goal = '';
  Future<void> getData() async {
    QuerySnapshot querySnapshot = await collectionRef.get();
    final allData = querySnapshot.docs.map((doc) => doc.data()).toList();
    dataFromDB = allData;
    print('In getData method ${dataFromDB.length} $dataFromDB');
    FirebaseAuth auth = FirebaseAuth.instance;
    final String? userEmail = auth.currentUser?.email.toString();
    for (final user in dataFromDB) {
      if (user["email"] == userEmail) {
        bmi = double.parse(user["bmi"]);
        goal = user["goal"];
      }
    }
  }

  @override
  void initState() {
    super.initState();
    getData();
  }

  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            margin: EdgeInsets.all(
              mediaQuery.size.height * 0.02,
            ),
            child: Image.asset('images/deadlifhome.jpg',
                height: mediaQuery.size.height * 0.4,
                width: mediaQuery.size.width * 0.6),
          ),
          Padding(
            padding: EdgeInsets.only(
                left: mediaQuery.size.height * 0.1,
                right: mediaQuery.size.height * 0.03),
            child: Text(
              'Recommended workout plans',
              style: TextStyle(
                color: kLightBlue,
                fontSize: mediaQuery.size.height * 0.05,
                fontFamily: 'Northwell',
              ),
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextButton(
                style: const ButtonStyle(
                  backgroundColor: MaterialStatePropertyAll(Colors.green),
                  foregroundColor: MaterialStatePropertyAll(Colors.white),
                ),
                onPressed: () {
                  Navigator.push(context, MaterialPageRoute(builder: ((context)=>WorkoutPlanPage(bmi: bmi))));
                },
                child: Text(
                  'Based on Your BMI',
                  style: TextStyle(
                    fontSize: mediaQuery.size.height * 0.02,
                    fontFamily: 'Vercetti',
                  ),
                ),
              ),
              SizedBox(
                width: mediaQuery.size.height*0.03,
              ),
              TextButton(
                style: const ButtonStyle(
                  backgroundColor: MaterialStatePropertyAll(Colors.green),
                  foregroundColor: MaterialStatePropertyAll(Colors.white),
                ),
                onPressed: () {
                  print(goal);
                  Navigator.push(context, MaterialPageRoute(builder: ((context)=>GoalBased(goal: goal))));
                },
                child: Text(
                  'Based on Your Goal',
                  style: TextStyle(
                    fontSize: mediaQuery.size.height * 0.02,
                    fontFamily: 'Vercetti',
                  ),
                ),
              )
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextButton(
                style: const ButtonStyle(
                  backgroundColor: MaterialStatePropertyAll(Colors.green),
                  foregroundColor: MaterialStatePropertyAll(Colors.white),
                ),
                onPressed: () {
                  Navigator.pushNamed(context, 'startWorkout');
                },
                child: Text(
                  'Start Custom Workout',
                  style: TextStyle(
                    fontSize: mediaQuery.size.height * 0.02,
                    fontFamily: 'Vercetti',
                  ),
                ),
              ),
              SizedBox(
                width: mediaQuery.size.height*0.03,
              ),
              TextButton(
                style: const ButtonStyle(
                  backgroundColor: MaterialStatePropertyAll(Colors.green),
                  foregroundColor: MaterialStatePropertyAll(Colors.white),
                ),
                onPressed: () {
                  Navigator.pushNamed(context, 'statistics');
                },
                child: Text(
                  'See Workout History',
                  style: TextStyle(
                    fontSize: mediaQuery.size.height * 0.02,
                    fontFamily: 'Vercetti',
                  ),
                ),
              )
            ],
          ),
          // Container(
          //   height: mediaQuery.size.height*0.4,
          //   width: double.infinity,
          //   child: WorkoutPlanPage(bmi: bmi),
          // ),
        ],
      ),
    );
  }
}

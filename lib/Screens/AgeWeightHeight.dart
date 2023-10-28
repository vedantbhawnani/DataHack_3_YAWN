import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';
import 'package:exerciserecognition/Components/customTextField.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class AgeWeight extends StatefulWidget {
  Map<String, String> request;
  AgeWeight({required this.request, super.key});

  @override
  State<AgeWeight> createState() => _AgeWeightState();
}

class _AgeWeightState extends State<AgeWeight> {
  TextEditingController ageController = TextEditingController();

  TextEditingController weightController = TextEditingController();

  TextEditingController heightController = TextEditingController();

  TextEditingController passwordController = TextEditingController();

  String dropDownItem = 'Gain Weight';

  var items = [
    'Gain Weight',
    'Stay Fit',
    'Lose Weight',
  ];

  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: [
            Center(
              child: Container(
                padding: EdgeInsets.only(top: mediaQuery.size.height * 0.1),
                child: Text(
                  'Create a new account',
                  style: TextStyle(
                    color: kBlackBackgroundColor,
                    fontSize: mediaQuery.size.height * 0.035,
                    fontFamily: 'ObjectSans',
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            Container(
              padding: EdgeInsets.all(mediaQuery.size.height * 0.03),
              child: Column(
                children: [
                  CustomTextField(
                    labelText: 'Age',
                    hintText: 'Enter Your Age',
                    controller: ageController,
                    obscureText: false,
                  ),
                  CustomTextField(
                    labelText: 'Weight',
                    hintText: 'Enter Your Weight (in kgs)',
                    controller: weightController,
                    obscureText: false,
                  ),
                  CustomTextField(
                    labelText: 'Height',
                    hintText: 'Enter Your Height (in cms)',
                    controller: heightController,
                    obscureText: false,
                  ),
                  Text(
                    'Your BMI will be calculated when you register',
                    style: TextStyle(
                      color: kLightBlue,
                      fontSize: mediaQuery.size.height * 0.02,
                    ),
                  ),
                  SizedBox(
                    height: mediaQuery.size.height*0.03,
                    width: double.infinity,
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Select Goal',
                        style: TextStyle(
                          color:kLightBlue,
                          fontSize: mediaQuery.size.height*0.02,
                          fontFamily: 'ObjectSans',
                        ),
                      ),
                      Container(
                        decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(mediaQuery.size.height*0.03,),
                            color: Colors.white,
                            border: Border.all(
                              color: kLightBlue,
                            )
                        ),
                        child: Padding(
                          padding: EdgeInsets.only(left: mediaQuery.size.height*0.06,right: mediaQuery.size.height*0.06,),
                          child: DropdownButton(
                              value: dropDownItem,
                              items: items.map((String items) {
                                return DropdownMenuItem(
                                  value: items,
                                  child: Text(items),
                                );
                              }).toList(),
                              onChanged: (String? newValue) {
                                setState(() {
                                  dropDownItem = newValue!;
                                });
                              }),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            SizedBox(
              width: mediaQuery.size.width * 0.5,
              height: mediaQuery.size.height * 0.07,
              child: TextButton(
                style: ButtonStyle(
                  elevation:
                      MaterialStatePropertyAll(mediaQuery.size.height * 0.04),
                  shadowColor:
                      const MaterialStatePropertyAll(kBlackBackgroundColor),
                  backgroundColor: const MaterialStatePropertyAll(kLightBlue),
                ),
                child: Text(
                  'Register',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: mediaQuery.size.height * 0.027,
                    fontFamily: 'ObjectSans',
                  ),
                ),
                onPressed: () {
                  widget.request['age'] = ageController.text;
                  widget.request['weight'] = weightController.text;
                  widget.request['height'] = heightController.text;
                  widget.request['goal']=dropDownItem;
                  int weight = int.parse(weightController.text);
                  int height = int.parse(heightController.text);
                  double finalHeight = height / 100;
                  double bmi = weight / (finalHeight * finalHeight);
                  widget.request['bmi'] = bmi.toString();
                  FirebaseFirestore db = FirebaseFirestore.instance;
                  FirebaseAuth auth = FirebaseAuth.instance;
                  DateTime uniqueId = DateTime.timestamp();
                  final String? userEmail = auth.currentUser?.email.toString();
                  String docName = '$userEmail $uniqueId';
                  print(widget.request);
                  db
                      .collection('User Details')
                      .doc(docName)
                      .set(widget.request);
                  Navigator.pushNamed(context, 'login');
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

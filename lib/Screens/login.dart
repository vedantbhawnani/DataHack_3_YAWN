import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';
import 'package:exerciserecognition/Components/customTextField.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../Service/auth.dart';

class LogIn extends StatefulWidget {
  const LogIn({super.key});

  @override
  State<LogIn> createState() => _LogInState();
}

class _LogInState extends State<LogIn> {
  TextEditingController emailController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
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
                padding: EdgeInsets.only(top: mediaQuery.size.height * 0.3),
                child: Text(
                  'Log In to your Account',
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
                    labelText: 'Email',
                    hintText: 'Enter Email Address',
                    controller: emailController,
                    obscureText: false,
                  ),
                  CustomTextField(
                    labelText: 'Password',
                    hintText: 'Enter Password',
                    controller: passwordController,
                    obscureText: true,
                  ),
                  SizedBox(
                    width: mediaQuery.size.width * 0.5,
                    height: mediaQuery.size.height * 0.07,
                    child: TextButton(
                      style: ButtonStyle(
                        elevation: MaterialStatePropertyAll(
                            mediaQuery.size.height * 0.04),
                        shadowColor: const MaterialStatePropertyAll(
                            kBlackBackgroundColor),
                        backgroundColor:
                            const MaterialStatePropertyAll(kLightBlue),
                      ),
                      child: Text(
                        'Log In',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: mediaQuery.size.height * 0.027,
                          fontFamily: 'ObjectSans',
                        ),
                      ),
                      onPressed: () {
                          try{
                            Auth().logInWithEmailAndPassword(email: emailController.text, password: passwordController.text);
                            if(FirebaseAuth.instance.currentUser!=null){
                              Navigator.pushNamed(context, 'homescreen');
                            }
                          } on FirebaseAuthException catch(e){
                          String text='Could not sign in';
                          if (e.code=='user-not-found'){
                             text='No user found for that email.';
                          }
                          else if(e.code=='wrong-password'){
                            text="Wrong password provided for that user.";
                          }
                          final snackBar = SnackBar(content: Text(text),);
                            ScaffoldMessenger.of(context).showSnackBar(snackBar);
                          }
                      },
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.all(mediaQuery.size.height * 0.04),
                    child: Column(
                      children: [
                        Text(
                          'Or Sign in using',
                          style: TextStyle(
                            fontFamily: 'Northwell',
                            fontSize: mediaQuery.size.height * 0.03,
                            color: kBlackBackgroundColor,
                          ),
                        ),
                        SizedBox(
                          height: mediaQuery.size.height * 0.02,
                          width: double.infinity,
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            TextButton(
                              style: ButtonStyle(
                                side: MaterialStatePropertyAll(BorderSide(width: 2.0,color: kLightBlue),),
                              ),
                              onPressed: () {},
                              child: Row(
                                children: [
                                  Image.asset(
                                    'images/google.png',
                                    height: mediaQuery.size.height * 0.05,
                                    width: mediaQuery.size.width * 0.05,
                                  ),
                                  Text(
                                    'Login With Google',
                                    style: TextStyle(
                                      color: kLightBlue,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            SizedBox(
                              width: mediaQuery.size.height*0.04,
                            ),
                            TextButton(
                              style: ButtonStyle(
                                side: MaterialStatePropertyAll(BorderSide(width: 2.0,color: kLightBlue),),
                              ),
                              onPressed: () {

                              },
                              child: Row(
                                children: [
                                  Image.asset(
                                    'images/meta.png',
                                    height: mediaQuery.size.height * 0.05,
                                    width: mediaQuery.size.width * 0.05,
                                  ),
                                  Text(
                                    'Login With Facebook',
                                    style: TextStyle(
                                      color: kLightBlue,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

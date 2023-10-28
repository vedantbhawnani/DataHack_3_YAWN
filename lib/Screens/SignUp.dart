import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';
import 'package:exerciserecognition/Components/customTextField.dart';
import 'package:exerciserecognition/Service/auth.dart';
import 'package:exerciserecognition/Screens/AgeWeightHeight.dart';


class SignUp extends StatefulWidget {
  const SignUp({super.key});

  @override
  State<SignUp> createState() => _SignUpState();
}

class _SignUpState extends State<SignUp> {
  TextEditingController nameController = TextEditingController();
  TextEditingController emailController = TextEditingController();
  TextEditingController phoneController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  TextEditingController confirmPasswordController = TextEditingController();
  TextEditingController occupationController = TextEditingController();
  String email='';
  String password ='';
  List dataList = [];
  Map<String,String> request = {
    'name':'some',
  };

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
                    labelText: 'Name',
                    hintText: 'Enter Full Name',
                    controller: nameController,
                    obscureText: false,
                  ),
                  CustomTextField(
                    labelText: 'Email',
                    hintText: 'Enter Email Address',
                    controller: emailController,
                    obscureText: false,
                  ),
                  CustomTextField(
                    labelText: 'Phone Number',
                    hintText: 'Enter Phone Number',
                    controller: phoneController,
                    obscureText: false,
                  ),
                  CustomTextField(
                    labelText: 'Password',
                    hintText: 'Enter Password',
                    controller: passwordController,
                    obscureText: true,
                  ),
                  CustomTextField(
                    labelText: 'Confirm Password',
                    hintText: 'Re-enter Password',
                    controller: confirmPasswordController,
                    obscureText: true,
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
                  shadowColor: const MaterialStatePropertyAll(kBlackBackgroundColor),
                  backgroundColor: const MaterialStatePropertyAll(kLightBlue),
                ),
                child: Text(
                  'Create User',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: mediaQuery.size.height * 0.027,
                    fontFamily: 'ObjectSans',
                  ),
                ),
                onPressed: () async {
                  email = emailController.text;
                  password = confirmPasswordController.text;
                  String text='User registered successfully, please continue';
                  try{
                    Auth().signUpWithEmailAndPassword(email: email, password: password);
                    final snackBar = SnackBar(content: Text(text),);
                    ScaffoldMessenger.of(context).showSnackBar(snackBar);
                  } on FirebaseAuthException catch(e){
                    if (e.code == 'weak-password'){
                      text='Password provided is too weak';
                    }
                    else if(e.code=='email-already-in-use'){
                      text="Email already registered, please sign in";
                    }
                    final snackBar = SnackBar(content: Text(text),);
                    ScaffoldMessenger.of(context).showSnackBar(snackBar);
                  }
                },
              ),
            ),
            SizedBox(
              height: mediaQuery.size.height*0.04,
              width: double.infinity,
            ),
            SizedBox(
              width: mediaQuery.size.width * 0.5,
              height: mediaQuery.size.height * 0.07,
              child: TextButton(
                style: ButtonStyle(
                  elevation:
                  MaterialStatePropertyAll(mediaQuery.size.height * 0.04),
                  shadowColor: const MaterialStatePropertyAll(kBlackBackgroundColor),
                  backgroundColor: const MaterialStatePropertyAll(kLightBlue),
                ),
                child: Text(
                  'Continue',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: mediaQuery.size.height * 0.027,
                    fontFamily: 'ObjectSans',
                  ),
                ),
                onPressed: (){
                  request["name"]=nameController.text;
                  request["email"]=emailController.text;
                  request["phoneNumber"]=phoneController.text;

                  Navigator.push(
                    context,MaterialPageRoute(builder: (context)=>AgeWeight(request: request))
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


import 'package:flutter/material.dart';
import 'package:exerciserecognition/constants.dart';

class CustomTextField extends StatelessWidget {
  final String labelText;
  final String hintText;
  final bool obscureText;
  final TextEditingController controller;

  const CustomTextField({required this.labelText,required this.hintText,required this.controller,required this.obscureText,super.key});

  @override
  Widget build(BuildContext context) {
    final mediaQuery = MediaQuery.of(context);
    return Container(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            labelText,
            style: TextStyle(
              color:kLightBlue,
              fontSize: mediaQuery.size.height*0.02,
              fontFamily: 'ObjectSans',
            ),
          ),
          SizedBox(
            width: double.infinity,
            height: mediaQuery.size.height*0.015,
          ),
          TextField(
            controller: controller,
            style: const TextStyle(color: kLightBlue),
            cursorColor: kLightBlue,
            decoration: kTextInputDecoration.copyWith(hintText: hintText),
            obscureText: obscureText,
          ),
          SizedBox(
            width: double.infinity,
            height: mediaQuery.size.height*0.02,
          ),
        ],
      ),
    );
  }
}

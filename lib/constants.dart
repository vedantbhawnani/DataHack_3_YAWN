import 'package:flutter/material.dart';

const kBlackBackgroundColor = Color(0xff02020A);
const kDarkBlue = Color(0xff05204A);
const kLightBlue = Color(0xff0094C6);
const kGrey = Colors.grey;

const kTextInputDecoration = InputDecoration(
  hintText: 'Enter a value',
  hintStyle: TextStyle(
    color: kBlackBackgroundColor,
  ),
  contentPadding:
  EdgeInsets.symmetric(vertical: 10.0, horizontal: 20.0),
  border: OutlineInputBorder(
    borderRadius: BorderRadius.all(Radius.circular(5.0)),
  ),
  enabledBorder: OutlineInputBorder(
    borderSide: BorderSide(color: kBlackBackgroundColor, width: 1.0),
    borderRadius: BorderRadius.all(Radius.circular(32.0)),
  ),
  focusedBorder: OutlineInputBorder(
    borderSide: BorderSide(color: kBlackBackgroundColor, width: 2.0),
    borderRadius: BorderRadius.all(Radius.circular(32.0)),
  ),
);
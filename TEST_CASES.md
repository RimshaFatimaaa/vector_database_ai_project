# Test Cases for AI Interview Coach - Answer Evaluation

## Overview
This document provides specific test cases to validate the answer evaluation system's functionality and accuracy.

## Test Case Categories

### 1. Behavioral Questions - Teamwork

#### Test Case 1.1: Excellent Teamwork Answer
**Question:** "Tell me about a time when you had to work in a team."

**Input:**
```
"In my previous role as a Senior Developer at TechCorp, I led a cross-functional team of 5 people to deliver a critical e-commerce platform migration. I organized daily standups, created a detailed project timeline, and ensured clear communication between frontend, backend, and DevOps teams. When we encountered a major integration issue, I facilitated a problem-solving session where each team member contributed their expertise. We successfully delivered the project 2 weeks ahead of schedule while maintaining 99.9% uptime."
```

**Expected Scores:**
- Overall Score: 90-95/100
- Relevance: 95-100/100
- Clarity: 90-95/100
- Correctness: 85-90/100

**Expected Feedback Elements:**
- Positive mention of leadership
- Specific examples and metrics
- Clear problem-solving approach
- Quantifiable results

#### Test Case 1.2: Poor Teamwork Answer
**Question:** "Tell me about a time when you had to work in a team."

**Input:**
```
"Umm I think I am good at teamwork, because in my last job I worked with a team of 5 people to build a Python application at Google. It was okay I guess."
```

**Expected Scores:**
- Overall Score: 25-40/100
- Relevance: 30-50/100
- Clarity: 20-40/100
- Correctness: 25-45/100

**Expected Feedback Elements:**
- Criticism of vagueness
- Lack of specific details
- Missing concrete examples
- Suggestions for improvement

### 2. Technical Questions - Programming

#### Test Case 2.1: Strong Technical Answer
**Question:** "Describe a challenging technical problem you solved."

**Input:**
```
"I faced a critical performance issue where our database queries were taking 15+ seconds. I analyzed the execution plans, identified missing indexes, and implemented query optimization. I also introduced Redis caching for frequently accessed data. The solution reduced query time to under 200ms and improved overall application performance by 75%. I documented the changes and created monitoring dashboards to prevent similar issues."
```

**Expected Scores:**
- Overall Score: 85-95/100
- Relevance: 90-95/100
- Clarity: 85-90/100
- Correctness: 90-95/100

**Expected Feedback Elements:**
- Technical depth and accuracy
- Problem-solving methodology
- Quantifiable results
- Proactive documentation

#### Test Case 2.2: Weak Technical Answer
**Question:** "Describe a challenging technical problem you solved."

**Input:**
```
"I had some problems with the code and I fixed them. It was hard but I figured it out. I used some tools and stuff."
```

**Expected Scores:**
- Overall Score: 15-30/100
- Relevance: 20-40/100
- Clarity: 10-25/100
- Correctness: 15-35/100

**Expected Feedback Elements:**
- Severe criticism of vagueness
- Lack of technical detail
- No specific solutions mentioned
- Strong suggestions for improvement

### 3. Edge Cases

#### Test Case 3.1: Empty Response
**Input:** ""

**Expected Behavior:**
- Error message: "Please enter a response to analyze."
- No evaluation performed
- User prompted to provide input

#### Test Case 3.2: Single Word Response
**Input:** "Yes"

**Expected Scores:**
- Overall Score: 5-15/100
- All scores very low
- Feedback emphasizing need for detail

#### Test Case 3.3: Very Long Response (500+ words)
**Input:** [Very long, rambling response with repetition]

**Expected Behavior:**
- System should handle gracefully
- May have lower clarity scores due to length
- Should still provide evaluation

#### Test Case 3.4: Non-English Response
**Input:** "Je suis un bon développeur avec beaucoup d'expérience."

**Expected Behavior:**
- System should attempt evaluation
- May have lower scores due to language mismatch
- Should provide feedback in English

### 4. Difficulty Level Testing

#### Test Case 4.1: Easy Question Response
**Question:** "What programming languages do you know?" (Easy)
**Input:** "I know Python, JavaScript, and Java."

**Expected Behavior:**
- Appropriate scoring for basic answer
- Feedback should be encouraging for entry-level response

#### Test Case 4.2: Hard Question Response
**Question:** "Explain the CAP theorem and its implications for distributed systems design." (Hard)
**Input:** "I know Python, JavaScript, and Java."

**Expected Behavior:**
- Low relevance score (doesn't address the question)
- Feedback should point out the mismatch

### 5. Sentiment Analysis Testing

#### Test Case 5.1: Positive Sentiment
**Input:** "I love working with technology and I'm passionate about solving complex problems. I enjoy collaborating with teams and learning new technologies."

**Expected Behavior:**
- Positive sentiment detected
- May contribute to higher clarity scores

#### Test Case 5.2: Negative Sentiment
**Input:** "I hate coding and I don't really want to work in tech. I'm just here because I need a job."

**Expected Behavior:**
- Negative sentiment detected
- May impact overall scores
- Feedback should address attitude concerns

### 6. Named Entity Recognition Testing

#### Test Case 6.1: With Company Names
**Input:** "I worked at Google for 3 years, then moved to Microsoft where I led a team of 10 developers."

**Expected Behavior:**
- Should recognize "Google" and "Microsoft" as organizations
- May contribute to higher correctness scores

#### Test Case 6.2: With Technical Terms
**Input:** "I used React, Node.js, and MongoDB to build a full-stack application with AWS deployment."

**Expected Behavior:**
- Should recognize technical terms
- May contribute to technical correctness scoring

### 7. Filler Word Detection

#### Test Case 7.1: High Filler Content
**Input:** "Umm, so I think, like, I'm pretty good at, you know, programming and stuff. I mean, I've done some things with, like, computers and things."

**Expected Behavior:**
- Should detect multiple filler words
- Should apply penalty to clarity score
- Feedback should mention reducing filler words

#### Test Case 7.2: Clean Response
**Input:** "I have five years of experience developing web applications using modern JavaScript frameworks and cloud technologies."

**Expected Behavior:**
- Minimal filler words detected
- Should not apply significant penalties
- Higher clarity scores

### 8. Repetition Detection

#### Test Case 8.1: High Repetition
**Input:** "I think I think I think I'm good at programming programming programming and I think I can do the job job job."

**Expected Behavior:**
- Should detect high repetition
- Should apply penalty to clarity score
- Feedback should suggest reducing repetition

### 9. Question-Answer Relevance Testing

#### Test Case 9.1: Highly Relevant
**Question:** "Tell me about your leadership experience."
**Input:** "I led a team of 8 developers for 2 years, managing project timelines and mentoring junior developers."

**Expected Behavior:**
- High relevance score
- Directly addresses the question

#### Test Case 9.2: Irrelevant
**Question:** "Tell me about your leadership experience."
**Input:** "I like pizza and my favorite color is blue. I have a dog named Max."

**Expected Behavior:**
- Very low relevance score
- Feedback should point out irrelevance

### 10. Performance Testing

#### Test Case 10.1: Response Time
**Input:** [Any valid response]

**Expected Behavior:**
- Evaluation should complete within 10-15 seconds
- Loading spinner should be displayed
- No timeout errors

#### Test Case 10.2: Concurrent Users
**Expected Behavior:**
- System should handle multiple simultaneous evaluations
- No crashes or errors
- Consistent performance

## Test Execution Guidelines

### Manual Testing Steps
1. Start the application
2. Log in with valid credentials
3. Generate a question
4. Input test case response
5. Click "Analyze Response"
6. Verify scores match expected ranges
7. Check feedback quality
8. Validate suggestions are helpful

### Automated Testing
- Unit tests for individual functions
- Integration tests for NLP + LLM pipeline
- Performance tests for response times
- Error handling tests for edge cases

### Regression Testing
- Run all test cases after any code changes
- Verify no existing functionality is broken
- Check performance hasn't degraded

---

**Version:** 1.0  
**Last Updated:** September 25, 2025  
**Author:** AI Interview Coach Development Team

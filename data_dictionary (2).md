# 📖 Data Dictionary — EduPro Online Platform Dataset

**File:** `EduPro_Online_Platform.xlsx`
**Sheets:** `Users`, `Teachers`, `Courses`, `Transactions`

This dictionary describes every field in the raw dataset as delivered — including `Teachers`, which is not currently used by the segmentation/recommendation model but is documented here for completeness.

---

## 1. Users
**3,000 rows | 5 columns** — one row per registered learner.

| Field | Type | Description | Notes |
|---|---|---|---|
| `UserID` | Text | Unique identifier for each learner | Format `U#####` (e.g. `U00001`). Primary key. |
| `UserName` | Text | Learner's platform username | 3,000 unique values (no duplicates). |
| `Age` | Integer | Learner's age in years | Range: 15–35. |
| `Gender` | Text (categorical) | Learner's gender | Values: `Male`, `Female`. |
| `Email` | Text | Learner's email address | 3,000 unique values. |

---

## 2. Teachers
**60 rows | 7 columns** — one row per course instructor. *(Not currently used in the segmentation/recommendation model — available for future features such as "instructor quality" signals.)*

| Field | Type | Description | Notes |
|---|---|---|---|
| `TeacherID` | Text | Unique identifier for each teacher | Format `TC#####` (e.g. `TC00001`). Primary key. |
| `TeacherName` | Text | Teacher's full name | 60 unique values. |
| `Age` | Integer | Teacher's age in years | Range: 27–50. |
| `Gender` | Text (categorical) | Teacher's gender | Values: `Male`, `Female`. |
| `Expertise` | Text (categorical) | Teacher's subject specialisation | 12 categories (same set as `CourseCategory` below): Artificial Intelligence, Business, Cybersecurity, Data Science, Design, Digital Marketing, Finance, Machine Learning, Marketing, Programming, Project Management, Web Development. |
| `YearsOfExperience` | Integer | Years of teaching/industry experience | Range: 1–24. |
| `TeacherRating` | Float | Average rating given to the teacher | Range: 1.05–4.97 (scale of 1–5). |

---

## 3. Courses
**60 rows | 8 columns** — one row per course offered on the platform.

| Field | Type | Description | Notes |
|---|---|---|---|
| `CourseID` | Text | Unique identifier for each course | Format `CR#####` (e.g. `CR00001`). Primary key. |
| `CourseName` | Text | Course title | 58 unique values (2 titles repeat, e.g. same course name offered at different levels). |
| `CourseCategory` | Text (categorical) | Subject area of the course | 12 categories: Artificial Intelligence, Business, Cybersecurity, Data Science, Design, Digital Marketing, Finance, Machine Learning, Marketing, Programming, Project Management, Web Development. |
| `CourseType` | Text (categorical) | Whether the course is paid or free | Values: `Paid`, `Free`. |
| `CourseLevel` | Text (categorical) | Difficulty level | Values: `Beginner`, `Intermediate`, `Advanced`. |
| `CoursePrice` | Float | Listed price of the course | Range: 0.0–490.9. Always 0.0 for `Free` courses. |
| `CourseDuration` | Float | Total course length, in hours | Range: 1.2–49.73. |
| `CourseRating` | Float | Average learner rating of the course | Range: 1.13–4.94 (scale of 1–5). |

---

## 4. Transactions
**10,000 rows | 7 columns** — one row per course purchase/enrolment event. This is the "fact" table that links `Users`, `Courses`, and `Teachers` together.

| Field | Type | Description | Notes |
|---|---|---|---|
| `TransactionID` | Text | Unique identifier for each transaction | Format `TT#####` (e.g. `TT00001`). Primary key. |
| `UserID` | Text | Learner who made the purchase | Foreign key → `Users.UserID`. All 10,000 values verified to exist in `Users`. |
| `CourseID` | Text | Course that was purchased | Foreign key → `Courses.CourseID`. All 10,000 values verified to exist in `Courses`. |
| `TransactionDate` | Date | Date the transaction occurred | Range: 2025-01-01 to 2025-12-30. |
| `Amount` | Float | Amount actually paid | Range: 0.0–490.9. Always 0.0 when the purchased course's `CourseType` is `Free`. |
| `PaymentMethod` | Text (categorical) | Payment method used | Values: `Credit Card`, `PayPal`, `Bank Transfer`. |
| `TeacherID` | Text | Teacher associated with the purchased course's delivery | Foreign key → `Teachers.TeacherID`. All 10,000 values verified to exist in `Teachers`. |

---

## 🔗 Relationships

```
Users (1) ───< Transactions (many) >─── Courses (1)
                     │
                     └───< Transactions (many) >─── Teachers (1)
```

- One learner (`Users`) can have many transactions.
- One course (`Courses`) can appear in many transactions.
- One teacher (`Teachers`) can be linked to many transactions.
- Every `UserID`, `CourseID`, and `TeacherID` in `Transactions` was confirmed to have a matching record in its parent sheet — no orphaned foreign keys.

## ✅ Data Quality Summary
- **Missing values:** 0 across all four sheets.
- **Duplicate rows:** 0 across all four sheets.
- **Referential integrity:** 100% — every foreign key in `Transactions` resolves to a valid row in `Users`, `Courses`, and `Teachers`.

## 🧮 Fields Engineered for Modelling
The segmentation/recommendation pipeline derives these learner-level features from the raw tables above (see `EduPro_Student_Segmentation.ipynb`, Step 3):

| Derived Field | Built From | Description |
|---|---|---|
| `TotalCourses` | `Transactions` (count per `UserID`) | Number of courses a learner has purchased. |
| `TotalSpend` | `Transactions.Amount` (sum per `UserID`) | Total amount spent by the learner. |
| `AvgSpend` | `Transactions.Amount` (mean per `UserID`) | Learner's typical price point. |
| `AvgCourseRating` | `Courses.CourseRating` (mean of purchased courses) | Average quality of courses the learner chooses. |
| `FavouriteCategory` | `Courses.CourseCategory` (mode per learner) | The learner's most-purchased subject area. |
| `FavouriteLevel` | `Courses.CourseLevel` (mode per learner) | The learner's most-purchased difficulty level. |
| `FreeCourseRatio` | `Courses.CourseType` (share of `Free` per learner) | Proportion of a learner's purchases that were free — a price-sensitivity signal. |
| `Recency` | `Transactions.TransactionDate` (days since latest, per learner) | How recently the learner last made a purchase. |
| `Cluster` | K-Means output on the features above | The learner's assigned behavioural segment (0 to k-1). |

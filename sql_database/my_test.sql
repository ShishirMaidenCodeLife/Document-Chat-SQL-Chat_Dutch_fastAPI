CREATE TABLE inventory_itemcategory (
  id INTEGER PRIMARY KEY NOT NULL,
  category_name TEXT DEFAULT NULL,
  created_by INTEGER DEFAULT NULL,
  organization_id INTEGER DEFAULT NULL,
  updated_at TEXT NOT NULL,
  created_at TEXT NOT NULL
);

INSERT INTO inventory_itemcategory (id, category_name, created_by, organization_id, updated_at, created_at) VALUES
(1, 'menu', 1, 1, '2023-12-03 12:15:03.209761', '2023-12-03 12:15:03.209328'),
(2, 'A', 1, 1, '2023-12-03 14:11:15.509865', '2023-12-03 14:11:15.509668'),
(3, 'B', 1, 1, '2023-12-03 14:13:49.651146', '2023-12-03 14:13:49.650926'),
(5, 'laptop', 1, 2, '2023-12-05 10:57:10.113322', '2023-12-05 10:57:10.113101'),
(6, 'pc', 1, 1, '2023-12-05 11:03:47.190049', '2023-12-05 11:03:47.189922'),
(7, 'laptop', 1, 1, '2023-12-05 12:00:49.073863', '2023-12-05 12:00:49.073649'),
(8, 'burger', 1, 1, '2023-12-05 13:26:29.897530', '2023-12-05 13:26:29.897242');


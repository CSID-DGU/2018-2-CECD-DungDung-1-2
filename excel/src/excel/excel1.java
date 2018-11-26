package excel;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFCell;

public class excel1 {	
	public static void main(String[] args) {
		XSSFRow row;
		XSSFCell cell;
		
		String value[] = new String[100];
		String value1[] = new String[100];

		try {
			FileInputStream coursename = new FileInputStream("C:\\Users\\NA\\Desktop/1234.xlsx");
			
			XSSFWorkbook coursenamebook = new XSSFWorkbook(coursename);
			//sheet수 취득
			//int sheetCn = workbook.getNumberOfSheets();
			int sheetCn=1;
			System.out.println("sheet수 : " + sheetCn);

			//시트만큼 반복
			for(int cn = 0; cn < sheetCn; cn++){
				System.out.println("취득하는 sheet 이름 : " + coursenamebook.getSheetName(cn));
				
				//0번째 sheet 정보 취득
				XSSFSheet sheet = coursenamebook.getSheetAt(cn);
				
				//취득된 sheet에서 rows수 취득
				int rows = sheet.getPhysicalNumberOfRows();
				System.out.println(coursenamebook.getSheetName(cn) + " sheet의 row수 : " + rows);

				//취득된 row에서 취득대상 cell수 취득
				int cells = sheet.getRow(cn).getPhysicalNumberOfCells();
				System.out.println(coursenamebook.getSheetName(cn) + " sheet의 row에 취득대상 cell수 : " + cells);

				for (int r = 0; r < rows; r++) {
					row = sheet.getRow(r); // row 가져오기
					if (row != null) {
						for (int c = 0; c < cells; c++) {
							cell = row.getCell(c);
							if (cell != null) {								
								value[r] = "" + cell.getStringCellValue();								
								System.out.print(value[r] + "\t");								
							} else {
								System.out.print("[null]\t");
							}
						} // for(c) 문
						System.out.print("\n");
					}
				} // for(r) 문
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		XSSFRow row1;
		XSSFCell cell1;
		try {
			FileInputStream course = new FileInputStream("C:\\Users\\NA\\Desktop/123.xlsx");
			
			XSSFWorkbook coursebook = new XSSFWorkbook(course);
			//sheet수 취득
			//int sheetCn = workbook.getNumberOfSheets();
			int sheetCn=1;
			System.out.println("sheet수 : " + sheetCn);

			//시트만큼 반복
			for(int cn = 0; cn < sheetCn; cn++){
				System.out.println("취득하는 sheet 이름 : " + coursebook.getSheetName(cn));
				
				//0번째 sheet 정보 취득
				XSSFSheet sheet = coursebook.getSheetAt(cn);
				
				//취득된 sheet에서 rows수 취득
				int rows = sheet.getPhysicalNumberOfRows();
				System.out.println(coursebook.getSheetName(cn) + " sheet의 row수 : " + rows);

				//취득된 row에서 취득대상 cell수 취득
				int cells = sheet.getRow(cn).getPhysicalNumberOfCells();
				System.out.println(coursebook.getSheetName(cn) + " sheet의 row에 취득대상 cell수 : " + cells);

				for (int r = 0; r < rows; r++) {
					row1 = sheet.getRow(r); // row 가져오기
					if (row1 != null) {
						for (int c = 0; c < cells; c++) {
							cell1 = row1.getCell(c);
							if (cell1 != null) {								
								value1[r] = "" + cell1.getStringCellValue();
								value1[r] = value1[r].replace("[[강의명]]", value[r]);
								System.out.print(value1[r] + "\t");								
							} else {
								System.out.print("[null]\t");
							}
						} // for(c) 문
						System.out.print("\n");
					}
				} // for(r) 문
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}
}
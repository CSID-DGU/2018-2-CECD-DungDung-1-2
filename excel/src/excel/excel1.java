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
		
		String words[] = new String[100];
		String sentences[] = new String[100];

		//단어 배열 만들기
		try {
			//파일 불러오기
			FileInputStream word = new FileInputStream("C:\\Users\\NA\\Desktop/1234.xlsx");
			XSSFWorkbook wordbook = new XSSFWorkbook(word);
			
			//sheet수 취득
			//int sheetCn = workbook.getNumberOfSheets();
			int sheetCn=1;			
			System.out.println("sheet수 : " + sheetCn);

			//시트만큼 반복
			for(int cn = 0; cn < sheetCn; cn++){
				System.out.println("취득하는 sheet 이름 : " + wordbook.getSheetName(cn));
				//0번째 sheet 정보 취득
				XSSFSheet sheet = wordbook.getSheetAt(cn);
				
				//취득된 sheet에서 rows수 취득
				int rows = sheet.getPhysicalNumberOfRows();
				System.out.println(wordbook.getSheetName(cn) + " sheet의 row수 : " + rows);

				//취득된 row에서 취득대상 cell수 취득
				int cells = sheet.getRow(cn).getPhysicalNumberOfCells();
				System.out.println(wordbook.getSheetName(cn) + " sheet의 row에 취득대상 cell수 : " + cells);

				for (int r = 0; r < rows; r++) {
					row = sheet.getRow(r); // row 가져오기
					if (row != null) {
						for (int c = 0; c < cells; c++) {
							cell = row.getCell(c);
							if (cell != null) {								
								words[r] = "" + cell.getStringCellValue();								
								System.out.print(words[r] + "\t");								
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
		
		//문장 배열 만들기
		XSSFRow row1;
		XSSFCell cell1;
		try {
			FileInputStream sentence = new FileInputStream("C:\\Users\\NA\\Desktop/123.xlsx");
			XSSFWorkbook sentencebook = new XSSFWorkbook(sentence);
			//sheet수 취득
			//int sheetCn = workbook.getNumberOfSheets();
			int sheetCn=1;
			System.out.println("sheet수 : " + sheetCn);

			//시트만큼 반복
			for(int cn = 0; cn < sheetCn; cn++){
				System.out.println("취득하는 sheet 이름 : " + sentencebook.getSheetName(cn));
				
				//0번째 sheet 정보 취득
				XSSFSheet sheet = sentencebook.getSheetAt(cn);
				
				//취득된 sheet에서 rows수 취득
				int rows = sheet.getPhysicalNumberOfRows();
				System.out.println(sentencebook.getSheetName(cn) + " sheet의 row수 : " + rows);

				//취득된 row에서 취득대상 cell수 취득
				int cells = sheet.getRow(cn).getPhysicalNumberOfCells();
				System.out.println(sentencebook.getSheetName(cn) + " sheet의 row에 취득대상 cell수 : " + cells);

				for (int r = 0; r < rows; r++) {
					row1 = sheet.getRow(r); // row 가져오기
					if (row1 != null) {
						for (int c = 0; c < cells; c++) {
							cell1 = row1.getCell(c);
							if (cell1 != null) {								
								sentences[r] = "" + cell1.getStringCellValue();
								//단어 바꾸기
								sentences[r] = sentences[r].replace("[[강의명]]", words[r]);
								System.out.print(sentences[r] + "\t");								
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